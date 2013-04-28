#include <vector>
using std::vector;

#include "Eigen/Core"
using Eigen::ArrayXXd;
using Eigen::MatrixXd;
using Eigen::Array;
using Eigen::Dynamic;

#include <iostream>
using std::cout;
using std::endl;

#include <cmath>
using std::min;
using std::ceil;

#include "utils.h"
#include "distribution.h"
#include "exception.h"
#include "conditionaldistribution.h"
#include "pcapreconditioner.h"
#include "tools.h"

typedef Array<bool, Dynamic, Dynamic> ArrayXXb;

template <class CD, class PC = CMT::PCAPreconditioner>
class PatchModel : public Distribution {
	public:
		typedef typename CD::Parameters Parameters;

		PatchModel(
			int rows,
			int cols,
			const ArrayXXb& inputMask,
			const ArrayXXb& outputMask,
			const CD* model = 0,
			int maxPCs = -1);
		virtual ~PatchModel();

		int dim() const;
		int rows() const;
		int cols() const;
		int maxPCs() const;
		ArrayXXb inputMask() const;
		ArrayXXb outputMask() const;

		CD& operator()(int i, int j);
		const CD& operator()(int i, int j) const;

		PC& preconditioner(int i, int j);
		const PC& preconditioner(int i, int j) const;
		void setPreconditioner(int i, int j, const PC& preconditioner);

		void initialize(const MatrixXd& data, const Parameters& params = Parameters());

		bool train(const MatrixXd& data, const Parameters& params = Parameters());
		bool train(
			const MatrixXd& data,
			const MatrixXd& dataVal,
			const Parameters& params = Parameters());

		Array<double, 1, Dynamic> logLikelihood(const MatrixXd& data) const;

		MatrixXd sample(int num_samples) const;

	protected:
		int mRows;
		int mCols;
		int mMaxPCs;
		ArrayXXb mInputMask;
		ArrayXXb mOutputMask;
		vector<Tuples> mInputIndices;
		vector<CD> mConditionalDistributions;
		vector<PC*> mPreconditioners;
};



template <class CD, class PC>
PatchModel<CD, PC>::PatchModel(
	int rows,
	int cols,
	const ArrayXXb& inputMask,
	const ArrayXXb& outputMask,
	const CD* model,
	int maxPCs) :
	mRows(rows),
	mCols(cols),
	mMaxPCs(maxPCs),
	mInputMask(inputMask),
	mOutputMask(outputMask)
{
	// compute locations of active pixels
	pair<Tuples, Tuples> inOutIndices = masksToIndices(inputMask, outputMask);
	Tuples& inputIndices = inOutIndices.first;
	Tuples& outputIndices = inOutIndices.second;

	if(outputIndices.size() > 1)
		throw Exception("Only one-dimensional outputs are currently supported.");

	if(model)
		if(inputIndices.size() != model->dimIn() || outputIndices.size() != model->dimOut())
			throw Exception("Model and masks are incompatible.");

	int rowOffset = outputIndices[0].first;
	int colOffset = outputIndices[0].second;

	for(Tuples::iterator it = inputIndices.begin(); it != inputIndices.end(); ++it)
		if(it->first > rowOffset || it->first == rowOffset && it->second > colOffset)
			throw Exception("Invalid masks. Only top-left to bottom-right sampling is currently supported.");

	// initialize conditional distributions with copy constructor
	for(int i = 0; i < rows; ++i)
		for(int j = 0; j < cols; ++j) {
			// compute indices of causal neighborhood at pixel (i, j)
			Tuples indices;

			for(Tuples::iterator it = inputIndices.begin(); it != inputIndices.end(); ++it) {
				// location of input pixel in patch
				int m = i + it->first - rowOffset;
				int n = j + it->second - colOffset;

				if(m >= 0 && m < mRows && n >= 0 && n < mCols)
					indices.push_back(make_pair(m, n));
			}

			mInputIndices.push_back(indices);

			// dimensionality of input
			int dimIn = mMaxPCs < 0 ? 
				indices.size() : min(static_cast<int>(indices.size()), mMaxPCs);

			// create model for pixel (i, j)
			if(model)
				if(dimIn == model->dimIn())
					// given model fits input
					mConditionalDistributions.push_back(*model);
				else
					// given model doesn't fit input
					mConditionalDistributions.push_back(CD(dimIn, *model));
			else
				// no model was given
				mConditionalDistributions.push_back(CD(dimIn));

			mPreconditioners.push_back(0);
		}
}



template <class CD, class PC>
PatchModel<CD, PC>::~PatchModel() {
	for(int i = 0; i < mRows * mCols; ++i)
		if(mPreconditioners[i])
			delete mPreconditioners[i];
}



template <class CD, class PC>
int PatchModel<CD, PC>::dim() const {
	return mRows * mCols;
}



template <class CD, class PC>
int PatchModel<CD, PC>::rows() const {
	return mRows;
}



template <class CD, class PC>
int PatchModel<CD, PC>::cols() const {
	return mCols;
}



template <class CD, class PC>
int PatchModel<CD, PC>::maxPCs() const {
	return mMaxPCs;
}



template <class CD, class PC>
ArrayXXb PatchModel<CD, PC>::inputMask() const {
	return mInputMask;
}



template <class CD, class PC>
ArrayXXb PatchModel<CD, PC>::outputMask() const {
	return mOutputMask;
}



template <class CD, class PC>
CD& PatchModel<CD, PC>::operator()(int i, int j) {
	if(i < 0 || j < 0 || j >= mCols || i >= mRows)
		throw Exception("Invalid indices.");
	return mConditionalDistributions[i * mCols + j];
}



template <class CD, class PC>
const CD& PatchModel<CD, PC>::operator()(int i, int j) const {
	if(i < 0 || j < 0 || j >= mCols || i >= mRows)
		throw Exception("Invalid indices.");
	return mConditionalDistributions[i * mCols + j];
}



template <class CD, class PC>
PC& PatchModel<CD, PC>::preconditioner(int i, int j) {
	if(i < 0 || j < 0 || j >= mCols || i >= mRows)
		throw Exception("Invalid indices.");
	if(!mPreconditioners[i * mCols + j])
		throw Exception("The model at this pixel has no preconditioner.");
	return *mPreconditioners[i * mCols + j];
}



template <class CD, class PC>
const PC& PatchModel<CD, PC>::preconditioner(int i, int j) const {
	if(i < 0 || j < 0 || j >= mCols || i >= mRows)
		throw Exception("Invalid indices.");
	if(!mPreconditioners[i * mCols + j])
		throw Exception("The model at this pixel has no preconditioner.");
	return *mPreconditioners[i * mCols + j];
}



template <class CD, class PC>
void PatchModel<CD, PC>::setPreconditioner(int i, int j, const PC& preconditioner) {
	if(mMaxPCs < 0)
		return;

	if(i < 0 || j < 0 || j >= mCols || i >= mRows)
		throw Exception("Invalid indices.");

	int index = i * mCols + j;

	if(preconditioner.dimIn() != mInputIndices[index].size() ||
		preconditioner.dimInPre() != mConditionalDistributions[index].dimIn() ||
		preconditioner.dimOutPre() != mConditionalDistributions[index].dimOut())
		throw Exception("Preconditioner is incompatible with model.");

	PC* preconditionerOld = mPreconditioners[index];
	mPreconditioners[index] = new PC(preconditioner);
	
	if(preconditionerOld)
		delete preconditionerOld;
}



template <class CD, class PC>
void PatchModel<CD, PC>::initialize(const MatrixXd& data, const Parameters& params) {
	Tuples inputIndices = maskToIndices(mInputMask);

	// count how many pixels possess a complete neighborhood
	int counter = 0;
	for(int i = 0; i < mRows * mCols; ++i)
		counter += mInputIndices[i].size() == inputIndices.size();

	int numSamplesPerImage = ceil(2 * data.cols() / static_cast<double>(counter)) + .5;
	if(numSamplesPerImage > data.cols())
		numSamplesPerImage = data.cols();

	// collect inputs and outputs
	vector<MatrixXd> inputs;
	vector<MatrixXd> outputs;

	for(int i = 0; i < mRows * mCols; ++i) {
		MatrixXd output = data.row(i);
		MatrixXd input(mInputIndices[i].size(), data.cols());

		for(int j = 0; j < mInputIndices[i].size(); ++j) {
			// coordinates of j-th input to i-th model
			int m = mInputIndices[i][j].first;
			int n = mInputIndices[i][j].second;

			// assumes patch is stored in row-major order
			input.row(j) = data.row(m * mCols + n);
		}

		if(mMaxPCs >= 0) {
			if(mPreconditioners[i])
				// delete existing preconditioner
				delete mPreconditioners[i];
			// create preconditioner
			mPreconditioners[i] = new PC(input, output, 0., mMaxPCs);
		}

		// check whether causal neighboor fits completely into image patch
		if(mInputIndices[i].size() == inputIndices.size()) {
			// keep a more or less random subset of the data for later
			int offset = rand() % (data.cols() - numSamplesPerImage + 1);

			inputs.push_back(input.block(0, offset, input.rows(), numSamplesPerImage));
			outputs.push_back(output.block(0, offset, output.rows(), numSamplesPerImage));

		} else {
			// initialize models with an incomplete neighborhood
			if(mMaxPCs < 0)
				mConditionalDistributions[i].initialize(input, output);
			else
				mConditionalDistributions[i].initialize(
					mPreconditioners[i]->operator()(input, output));
		}
	}

	MatrixXd input = concatenate(inputs);
	MatrixXd output = concatenate(outputs);

	// number of training and validation data points
	int numTrain = ceil(input.cols() * 0.9) + .5;
	int numValid = input.cols() - numTrain;

	if(!numValid)
		// too few data points
		throw Exception("Too few data points.");

	for(int i = 0; i < mRows * mCols; ++i)
		if(mInputIndices[i].size() == inputIndices.size()) {
			// train a single model
			if(mMaxPCs < 0) {
				mConditionalDistributions[i].initialize(input, output);
				mConditionalDistributions[i].train(
					// training set
					input.leftCols(numTrain),
					output.leftCols(numTrain),
					// validation set
					input.rightCols(numValid),
					output.rightCols(numValid),
					params);
			} else {
				// compute preconditioned input and output
				PC pc(input, output, 0., mMaxPCs);
				pair<ArrayXXd, ArrayXXd> data = pc(input, output);
				const MatrixXd& inputPc = data.first;
				const MatrixXd& outputPc = data.second;

				mConditionalDistributions[i].initialize(inputPc, outputPc);
				mConditionalDistributions[i].train(
					// training set
					inputPc.leftCols(numTrain),
					outputPc.leftCols(numTrain),
					// validation set
					inputPc.rightCols(numValid),
					outputPc.rightCols(numValid),
					params);
			}

			// copy parameters to all other models with the same input size
			for(int j = i + 1; j < mRows * mCols; ++j)
				if(mConditionalDistributions[j].dimIn() == mConditionalDistributions[i].dimIn())
					mConditionalDistributions[j] = mConditionalDistributions[i];

			break;
		}
}



template <class CD, class PC>
bool PatchModel<CD, PC>::train(const MatrixXd& data, const Parameters& params) {
	bool converged = true;

	for(int i = 0; i < mRows * mCols; ++i) {
		// assumes patch is stored in row-major order
		MatrixXd output = data.row(i);
		MatrixXd input(mInputIndices[i].size(), data.cols());

		// extract inputs and outputs from patches
		#pragma omp parallel for
		for(int j = 0; j < mInputIndices[i].size(); ++j) {
			// coordinates of j-th input to i-th model
			int m = mInputIndices[i][j].first;
			int n = mInputIndices[i][j].second;

			// assumes patch is stored in row-major order
			input.row(j) = data.row(m * mCols + n);
		}

		if(params.verbosity > 0)
			cout << "Training model " << i / mCols << ", " << i % mCols << endl;

		if(mMaxPCs < 0) {
			converged &= mConditionalDistributions[i].train(input, output, params);
		} else {
			if(!mPreconditioners[i])
				throw Exception("Model has to be initialized first.");
			converged &= mConditionalDistributions[i].train(
				mPreconditioners[i]->operator()(input, output), params);
		}
	}

	return converged;
}



template <class CD, class PC>
bool PatchModel<CD, PC>::train(
	const MatrixXd& data,
	const MatrixXd& dataVal,
	const Parameters& params)
{
	bool converged = true;

	for(int i = 0; i < mRows * mCols; ++i) {
		// assumes patch is stored in row-major order
		MatrixXd output = data.row(i);
		MatrixXd input(mInputIndices[i].size(), data.cols());
		MatrixXd outputVal = dataVal.row(i);
		MatrixXd inputVal(mInputIndices[i].size(), dataVal.cols());

		// extract inputs and outputs from patches
		#pragma omp parallel for
		for(int j = 0; j < mInputIndices[i].size(); ++j) {
			// coordinates of j-th input to i-th model
			int m = mInputIndices[i][j].first;
			int n = mInputIndices[i][j].second;

			// assumes patch is stored in row-major order
			input.row(j) = data.row(m * mCols + n);
			inputVal.row(j) = dataVal.row(m * mCols + n);
		}

		if(params.verbosity > 0)
			cout << "Training model " << i / mCols << ", " << i % mCols << endl;

		if(mMaxPCs < 0) {
			converged &= mConditionalDistributions[i].train(
				input, output, inputVal, outputVal, params);
		} else {
			if(!mPreconditioners[i])
				throw Exception("Model has to be initialized first.");
			converged &= mConditionalDistributions[i].train(
				mPreconditioners[i]->operator()(input, output),
				mPreconditioners[i]->operator()(inputVal, outputVal),
				params);
		}
	}

	return converged;
}



template <class CD, class PC>
Array<double, 1, Dynamic> PatchModel<CD, PC>::logLikelihood(
	const MatrixXd& data) const 
{
	Array<double, 1, Dynamic> logLik = Array<double, 1, Dynamic>::Zero(data.cols());

	for(int i = 0; i < mRows * mCols; ++i) {
		// assumes patch is stored in row-major order
		MatrixXd output = data.row(i);
		MatrixXd input(mInputIndices[i].size(), data.cols());

		for(int j = 0; j < mInputIndices[i].size(); ++j) {
			// coordinates of j-th input to i-th model
			int m = mInputIndices[i][j].first;
			int n = mInputIndices[i][j].second;

			// assumes patch is stored in row-major order
			input.row(j) = data.row(m * mCols + n);
		}

		if(mMaxPCs < 0) {
			logLik += mConditionalDistributions[i].logLikelihood(input, output);
		} else {
			if(!mPreconditioners[i])
				throw Exception("Model has to be initialized first.");
			logLik += mConditionalDistributions[i].logLikelihood(
				mPreconditioners[i]->operator()(input, output));
			logLik += mPreconditioners[i]->logJacobian(input, output);
		}
	}

	return logLik;
}



template<class CD, class PC>
MatrixXd PatchModel<CD, PC>::sample(int num_samples) const {
	MatrixXd samples = MatrixXd::Zero(mRows * mCols, num_samples);

	for(int i = 0; i < mRows * mCols; ++i) {
		MatrixXd input(mInputIndices[i].size(), num_samples);

		// construct input from already sampled patch
		for(int j = 0; j < mInputIndices[i].size(); ++j) {
			// coordinates of j-th input to i-th model
			int m = mInputIndices[i][j].first;
			int n = mInputIndices[i][j].second;

			// assumes patch is stored in row-major order
			input.row(j) = samples.row(m * mCols + n);
		}

		if(mMaxPCs < 0) {
			samples.row(i) = mConditionalDistributions[i].sample(input);
		} else {
			if(!mPreconditioners[i])
				throw Exception("Model has to be initialized first.");
			MatrixXd inputPc = mPreconditioners[i]->operator()(input);
			MatrixXd outputPc = mConditionalDistributions[i].sample(inputPc);
			samples.row(i) = mPreconditioners[i]->inverse(inputPc, outputPc).second;
		}
	}

	return samples;
}
