#ifndef CMT_PATCHMODEL_H
#define CMT_PATCHMODEL_H

#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>
#include "Eigen/Core"
#include "utils.h"
#include "distribution.h"
#include "exception.h"
#include "conditionaldistribution.h"
#include "pcapreconditioner.h"
#include "tools.h"

namespace CMT {
	using std::vector;

	using std::find;
	using std::distance;

	using std::cout;
	using std::endl;

	using std::min;
	using std::ceil;

	using Eigen::ArrayXXd;
	using Eigen::MatrixXd;
	using Eigen::Array;
	using Eigen::Dynamic;

	class PatchModelBase : public Distribution {
		public:
			using Distribution::logLikelihood;

			virtual ~PatchModelBase();

			virtual int rows() const = 0;
			virtual int cols() const = 0;
			virtual int maxPCs() const = 0;
			virtual ArrayXXb inputMask() const = 0;
			virtual ArrayXXb outputMask() const = 0;
			virtual ArrayXXb inputMask(int i, int j) const = 0;
			virtual ArrayXXb outputMask(int i, int j) const = 0;
			virtual Tuples inputIndices(int i, int j) const = 0;
			virtual Tuples order() const = 0;

			virtual Array<double, 1, Dynamic> logLikelihood(
				int i, int j, const MatrixXd& data) const = 0;
	};

	template <class CD, class PC = CMT::PCAPreconditioner>
	class PatchModel : public PatchModelBase {
		public:
			typedef typename CD::Parameters Parameters;

			PatchModel(
				int rows,
				int cols,
				const CD* model = 0,
				int maxPCs = -1);
			PatchModel(
				int rows,
				int cols,
				const ArrayXXb& inputMask,
				const ArrayXXb& outputMask,
				const CD* model = 0,
				int maxPCs = -1);
			PatchModel(
				int rows,
				int cols,
				const Tuples& order,
				const ArrayXXb& inputMask,
				const ArrayXXb& outputMask,
				const CD* model = 0,
				int maxPCs = -1);
			PatchModel(
				int rows,
				int cols,
				const Tuples& order,
				const CD* model = 0,
				int maxPCs = -1);
			virtual ~PatchModel();

			int dim() const;
			int rows() const;
			int cols() const;
			int maxPCs() const;
			ArrayXXb inputMask() const;
			ArrayXXb outputMask() const;
			ArrayXXb inputMask(int i, int j) const;
			ArrayXXb outputMask(int i, int j) const;
			Tuples inputIndices(int i, int j) const;
			Tuples order() const;

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
			bool train(
				int i,
				int j,
				const MatrixXd& data,
				const Parameters& params = Parameters());
			bool train(
				int i,
				int j,
				const MatrixXd& data,
				const MatrixXd& dataVal,
				const Parameters& params = Parameters());

			Array<double, 1, Dynamic> logLikelihood(const MatrixXd& data) const;
			Array<double, 1, Dynamic> logLikelihood(int i, int j, const MatrixXd& data) const;

			MatrixXd sample(int num_samples) const;

		protected:
			int mRows;
			int mCols;
			int mMaxPCs;
			ArrayXXb mInputMask;
			ArrayXXb mOutputMask;
			Tuples mOutputIndices;
			vector<Tuples> mInputIndices;
			vector<CD> mConditionalDistributions;
			vector<PC*> mPreconditioners;

			int findIndex(int i, int j) const;
	};
}



/**
 * Constructs model which assumes image patches are generated pixel by pixel
 * in row-major order.
 *
 * @param rows number of rows of the image patch
 * @param cols number of columns of the image patch
 * @param model template which will be used to initialize other models
 * @param maxPCs used to reduce dimensionality of input to each model via PCA first
 */
template <class CD, class PC>
CMT::PatchModel<CD, PC>::PatchModel(
	int rows,
	int cols,
	const CD* model,
	int maxPCs) :
	mRows(rows),
	mCols(cols),
	mMaxPCs(maxPCs)
{
	mInputMask = ArrayXXb::Ones(2 * rows - 1, 2 * cols - 1);
	mInputMask(rows - 1, cols - 1) = false;
	mOutputMask = ArrayXXb::Zero(2 * rows - 1, 2 * cols - 1);
	mOutputMask(rows - 1, cols - 1) = true;

	// initialize conditional distributions
	for(int i = 0; i < mRows; ++i)
		for(int j = 0; j < mCols; ++j) {
			// compute indices of input to model predicting pixel (i, j)
			Tuples indices;

			for(Tuples::iterator it = mOutputIndices.begin(); it != mOutputIndices.end(); ++it)
				indices.push_back(*it);

			mInputIndices.push_back(indices);
			mOutputIndices.push_back(make_pair(i, j));

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



/**
 * Constructs model which assumes image patches are generated pixel by pixel
 * in row-major order.
 *
 * @param rows number of rows of the image patch
 * @param cols number of columns of the image patch
 * @param inputMask mask used to constrain input to certain pixels
 * @param outputMask mask indicating the output pixel
 * @param model template which will be used to initialize other models
 * @param maxPCs used to reduce dimensionality of input to each model via PCA first
 */
template <class CD, class PC>
CMT::PatchModel<CD, PC>::PatchModel(
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

	int rowOffset = outputIndices[0].first;
	int colOffset = outputIndices[0].second;

	// initialize conditional distributions
	for(int i = 0; i < rows; ++i)
		for(int j = 0; j < cols; ++j) {
			// compute indices of causal neighborhood at pixel (i, j)
			Tuples indices;

			for(Tuples::iterator it = inputIndices.begin(); it != inputIndices.end(); ++it) {
				// location of input pixel in patch
				int m = i + it->first - rowOffset;
				int n = j + it->second - colOffset;

				if(m >= 0 && n >= 0 && m < mRows && n < mCols && m <= i && (n < j || m < i))
					indices.push_back(make_pair(m, n));
			}

			mInputIndices.push_back(indices);
			mOutputIndices.push_back(make_pair(i, j));

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



/**
 * Constructs model which generates pixels in a prespecified order.
 *
 * @param rows number of rows of the image patch
 * @param cols number of columns of the image patch
 * @param order list of pixel locations
 * @param inputMask mask used to constrain input to certain pixels
 * @param outputMask mask indicating the output pixel
 * @param model template which will be used to initialize other models
 * @param maxPCs used to reduce dimensionality of input to each model via PCA first
 */
template <class CD, class PC>
CMT::PatchModel<CD, PC>::PatchModel(
	int rows,
	int cols,
	const Tuples& order,
	const ArrayXXb& inputMask,
	const ArrayXXb& outputMask,
	const CD* model,
	int maxPCs) :
	mRows(rows),
	mCols(cols),
	mMaxPCs(maxPCs),
	mInputMask(inputMask),
	mOutputMask(outputMask),
	mOutputIndices(order)
{
	if(order.size() != mRows * mCols)
		throw Exception("Invalid pixel order.");

	// compute location of output index
	Tuples outputIndices = maskToIndices(outputMask);

	if(outputIndices.size() > 1)
		throw Exception("Only one-dimensional outputs are currently supported.");

	int rowOffset = outputIndices[0].first;
	int colOffset = outputIndices[0].second;

	// initialize conditional distributions
	for(Tuples::iterator oit = mOutputIndices.begin(); oit != mOutputIndices.end(); ++oit) {
		// location of pixel in patch
		int i = oit->first;
		int j = oit->second;

		if(i < 0 || j < 0 || i >= mRows || j >= mCols)
			throw Exception("Invalid pixel location in list of pixels.");

		// compute indices of input to model predicting pixel (i, j)
		Tuples indices;

		for(Tuples::iterator iit = mOutputIndices.begin(); iit != oit; ++iit) {
			// location of input pixel in mask
			int m = iit->first - i + rowOffset;
			int n = iit->second - j + colOffset;

			// check if pixel is active in input mask
			if(m >= 0 && n >= 0 && m < inputMask.rows() && n < inputMask.cols())
				if(inputMask(m, n))
					indices.push_back(*iit);
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
CMT::PatchModel<CD, PC>::PatchModel(
	int rows,
	int cols,
	const Tuples& order,
	const CD* model,
	int maxPCs) :
	mRows(rows),
	mCols(cols),
	mMaxPCs(maxPCs),
	mOutputIndices(order)
{
	if(order.size() != mRows * mCols)
		throw Exception("Invalid pixel order.");

	mInputMask = ArrayXXb::Ones(2 * rows - 1, 2 * cols - 1);
	mInputMask(rows - 1, cols - 1) = false;
	mOutputMask = ArrayXXb::Zero(2 * rows - 1, 2 * cols - 1);
	mOutputMask(rows - 1, cols - 1) = true;

	// initialize conditional distributions
	for(Tuples::iterator oit = mOutputIndices.begin(); oit != mOutputIndices.end(); ++oit) {
		// location of pixel in patch
		int i = oit->first;
		int j = oit->second;

		if(i < 0 || j < 0 || i >= mRows || j >= mCols)
			throw Exception("Invalid pixel location in list of pixels.");

		// compute indices of input to model predicting pixel (i, j)
		Tuples indices;

		for(Tuples::iterator iit = mOutputIndices.begin(); iit != oit; ++iit)
			indices.push_back(*iit);

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
CMT::PatchModel<CD, PC>::~PatchModel() {
	for(int i = 0; i < mRows * mCols; ++i)
		if(mPreconditioners[i])
			delete mPreconditioners[i];
}



template <class CD, class PC>
int CMT::PatchModel<CD, PC>::dim() const {
	return mRows * mCols;
}



template <class CD, class PC>
int CMT::PatchModel<CD, PC>::rows() const {
	return mRows;
}



template <class CD, class PC>
int CMT::PatchModel<CD, PC>::cols() const {
	return mCols;
}



template <class CD, class PC>
int CMT::PatchModel<CD, PC>::maxPCs() const {
	return mMaxPCs;
}



template <class CD, class PC>
Eigen::ArrayXXb CMT::PatchModel<CD, PC>::inputMask() const {
	return mInputMask;
}



template <class CD, class PC>
Eigen::ArrayXXb CMT::PatchModel<CD, PC>::outputMask() const {
	return mOutputMask;
}



template <class CD, class PC>
Eigen::ArrayXXb CMT::PatchModel<CD, PC>::inputMask(int i, int j) const {
	ArrayXXb inputMask = ArrayXXb::Zero(mRows, mCols);

	int k = findIndex(i, j);

	for(Tuples::const_iterator it = mInputIndices[k].begin(); it != mInputIndices[k].end(); ++it)
		inputMask(it->first, it->second) = true;

	return inputMask;
}



template <class CD, class PC>
Eigen::ArrayXXb CMT::PatchModel<CD, PC>::outputMask(int i, int j) const {
	ArrayXXb outputMask = ArrayXXb::Zero(mRows, mCols);
	outputMask(i, j) = true;
	return outputMask;
}



template <class CD, class PC>
CMT::Tuples CMT::PatchModel<CD, PC>::inputIndices(int i, int j) const {
	return mInputIndices[findIndex(i, j)];
}



template <class CD, class PC>
CMT::Tuples CMT::PatchModel<CD, PC>::order() const {
	return mOutputIndices;
}



template <class CD, class PC>
CD& CMT::PatchModel<CD, PC>::operator()(int i, int j) {
	if(i < 0 || j < 0 || j >= mCols || i >= mRows)
		throw Exception("Invalid indices.");
	return mConditionalDistributions[findIndex(i, j)];
}



template <class CD, class PC>
const CD& CMT::PatchModel<CD, PC>::operator()(int i, int j) const {
	if(i < 0 || j < 0 || j >= mCols || i >= mRows)
		throw Exception("Invalid indices.");
	return mConditionalDistributions[findIndex(i, j)];
}



template <class CD, class PC>
PC& CMT::PatchModel<CD, PC>::preconditioner(int i, int j) {
	if(i < 0 || j < 0 || j >= mCols || i >= mRows)
		throw Exception("Invalid indices.");

	int k = findIndex(i, j);

	if(!mPreconditioners[k])
		throw Exception("The model at this pixel has no preconditioner.");

	return *mPreconditioners[k];
}



template <class CD, class PC>
const PC& CMT::PatchModel<CD, PC>::preconditioner(int i, int j) const {
	if(i < 0 || j < 0 || j >= mCols || i >= mRows)
		throw Exception("Invalid indices.");

	int k = findIndex(i, j);

	if(!mPreconditioners[k])
		throw Exception("The model for this pixel has no preconditioner.");

	return *mPreconditioners[k];
}



template <class CD, class PC>
void CMT::PatchModel<CD, PC>::setPreconditioner(int i, int j, const PC& preconditioner) {
	if(mMaxPCs < 0)
		return;

	if(i < 0 || j < 0 || j >= mCols || i >= mRows)
		throw Exception("Invalid indices.");

	int k = findIndex(i, j);

	if(preconditioner.dimIn() != mInputIndices[k].size() ||
		preconditioner.dimInPre() != mConditionalDistributions[k].dimIn() ||
		preconditioner.dimOutPre() != mConditionalDistributions[k].dimOut())
		throw Exception("Preconditioner is incompatible with model.");

	PC* preconditionerOld = mPreconditioners[k];
	mPreconditioners[k] = new PC(preconditioner);
	
	if(preconditionerOld)
		delete preconditionerOld;
}



template <class CD, class PC>
void CMT::PatchModel<CD, PC>::initialize(const MatrixXd& data, const Parameters& params) {
	for(int i = 0; i < mRows * mCols; ++i) {
		int m = mOutputIndices[i].first;
		int n = mOutputIndices[i].second;

		// assumes patch is stored in row-major order
		MatrixXd output = data.row(m * mCols + n);
		MatrixXd input(mInputIndices[i].size(), data.cols());

		#pragma omp parallel for
		for(int j = 0; j < mInputIndices[i].size(); ++j) {
			// coordinates of j-th input to i-th model
			int m = mInputIndices[i][j].first;
			int n = mInputIndices[i][j].second;

			// assumes patch is stored in row-major order
			input.row(j) = data.row(m * mCols + n);
		}

		if(mMaxPCs >= 0) {
			if(mPreconditioners[i])
				delete mPreconditioners[i];
			mPreconditioners[i] = new PC(input, output, 0., mMaxPCs);
			mConditionalDistributions[i].initialize(
				mPreconditioners[i]->operator()(input, output));
		} else {
			mConditionalDistributions[i].initialize(input, output);
		}
	}
}



template <class CD, class PC>
bool CMT::PatchModel<CD, PC>::train(const MatrixXd& data, const Parameters& params) {
	bool converged = true;

	for(int i = 0; i < mRows * mCols; ++i) {
		// coordinates of i-th output pixels
		int m = mOutputIndices[i].first;
		int n = mOutputIndices[i].second;

		// assumes patch is stored in row-major order
		MatrixXd output = data.row(m * mCols + n);
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
				mPreconditioners[i] = new PC(input, output, 0., mMaxPCs);
			converged &= mConditionalDistributions[i].train(
				mPreconditioners[i]->operator()(input, output), params);
		}
	}

	return converged;
}



template <class CD, class PC>
bool CMT::PatchModel<CD, PC>::train(
	const MatrixXd& data,
	const MatrixXd& dataVal,
	const Parameters& params)
{
	bool converged = true;

	for(int i = 0; i < mRows * mCols; ++i) {
		// coordinates of i-th output pixels
		int m = mOutputIndices[i].first;
		int n = mOutputIndices[i].second;

		// assumes patch is stored in row-major order
		MatrixXd output = data.row(m * mCols + n);
		MatrixXd input(mInputIndices[i].size(), data.cols());
		MatrixXd outputVal = dataVal.row(m * mCols + n);
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
				mPreconditioners[i] = new PC(input, output, 0., mMaxPCs);
			converged &= mConditionalDistributions[i].train(
				mPreconditioners[i]->operator()(input, output),
				mPreconditioners[i]->operator()(inputVal, outputVal),
				params);
		}
	}

	return converged;
}



template <class CD, class PC>
bool CMT::PatchModel<CD, PC>::train(int i, int j, const MatrixXd& data, const Parameters& params) {
	int k = findIndex(i, j);

	MatrixXd output = data.row(i * mCols + j);
	MatrixXd input(mInputIndices[k].size(), data.cols());

	// extract inputs and outputs from patches
	#pragma omp parallel for
	for(int l = 0; l < mInputIndices[k].size(); ++l) {
		// coordinates of j-th input to i-th model
		int m = mInputIndices[k][l].first;
		int n = mInputIndices[k][l].second;

		// assumes patch is stored in row-major order
		input.row(l) = data.row(m * mCols + n);
	}

	if(mMaxPCs < 0) {
		return mConditionalDistributions[k].train(input, output, params);
	} else {
		if(!mPreconditioners[k])
			mPreconditioners[k] = new PC(input, output, 0., mMaxPCs);
		return mConditionalDistributions[k].train(
			mPreconditioners[k]->operator()(input, output), params);
	}
}



template <class CD, class PC>
bool CMT::PatchModel<CD, PC>::train(
	int i,
	int j,
	const MatrixXd& data,
	const MatrixXd& dataVal,
	const Parameters& params)
{
	int k = findIndex(i, j);

	// assumes patch is stored in row-major order
	MatrixXd output = data.row(i * mCols + j);
	MatrixXd input(mInputIndices[k].size(), data.cols());
	MatrixXd outputVal = dataVal.row(i * mCols + j);
	MatrixXd inputVal(mInputIndices[k].size(), dataVal.cols());

	// extract inputs and outputs from patches
	#pragma omp parallel for
	for(int l = 0; l < mInputIndices[k].size(); ++l) {
		// coordinates of j-th input to i-th model
		int m = mInputIndices[k][l].first;
		int n = mInputIndices[k][l].second;

		// assumes patch is stored in row-major order
		input.row(l) = data.row(m * mCols + n);
		inputVal.row(l) = dataVal.row(m * mCols + n);
	}

	if(mMaxPCs < 0) {
		return mConditionalDistributions[k].train(
			input, output, inputVal, outputVal, params);
	} else {
		if(!mPreconditioners[k])
			mPreconditioners[k] = new PC(input, output, 0., mMaxPCs);
		return mConditionalDistributions[k].train(
			mPreconditioners[k]->operator()(input, output),
			mPreconditioners[k]->operator()(inputVal, outputVal),
			params);
	}
}



template <class CD, class PC>
Eigen::Array<double, 1, Eigen::Dynamic> CMT::PatchModel<CD, PC>::logLikelihood(
	const MatrixXd& data) const
{
	if(data.rows() != dim())
		throw Exception("Data has wrong dimensionality.");

	if(mMaxPCs < 0)
		for(int i = 0; i < mRows * mCols; ++i)
			if(!mPreconditioners[i])
				throw Exception("Model has to be initialized first.");

	Array<double, 1, Dynamic> logLik = Array<double, 1, Dynamic>::Zero(data.cols());

	#pragma omp parallel for
	for(int i = 0; i < mRows * mCols; ++i) {
		int m = mOutputIndices[i].first;
		int n = mOutputIndices[i].second;

		// assumes patch is stored in row-major order
		MatrixXd output = data.row(m * mCols + n);
		MatrixXd input(mInputIndices[i].size(), data.cols());

		for(int j = 0; j < mInputIndices[i].size(); ++j) {
			// coordinates of j-th input to i-th model
			int m = mInputIndices[i][j].first;
			int n = mInputIndices[i][j].second;

			// assumes patch is stored in row-major order
			input.row(j) = data.row(m * mCols + n);
		}

		Array<double, 1, Dynamic> logLik_;

		if(mMaxPCs < 0)
			logLik_ = mConditionalDistributions[i].logLikelihood(input, output);
		else
			logLik_ = mConditionalDistributions[i].logLikelihood(
				mPreconditioners[i]->operator()(input, output)) +
				mPreconditioners[i]->logJacobian(input, output);

		#pragma omp critical
		logLik += logLik_;
	}

	return logLik;
}



template <class CD, class PC>
Eigen::Array<double, 1, Eigen::Dynamic> CMT::PatchModel<CD, PC>::logLikelihood(
	int i, int j, const MatrixXd& data) const
{
	if(data.rows() != dim())
		throw Exception("Data has wrong dimensionality.");

	int k = findIndex(i, j);

	Array<double, 1, Dynamic> logLik = Array<double, 1, Dynamic>::Zero(data.cols());

	// assumes patch is stored in row-major order
	MatrixXd output = data.row(i * mCols + j);
	MatrixXd input(mInputIndices[k].size(), data.cols());

	for(int l = 0; l < mInputIndices[k].size(); ++l) {
		// coordinates of j-th input to i-th model
		int m = mInputIndices[k][l].first;
		int n = mInputIndices[k][l].second;

		// assumes patch is stored in row-major order
		input.row(l) = data.row(m * mCols + n);
	}

	if(mMaxPCs < 0) {
		return mConditionalDistributions[k].logLikelihood(input, output);
	} else {
		if(!mPreconditioners[k])
			throw Exception("Model has to be initialized first.");
		return mConditionalDistributions[k].logLikelihood(
			mPreconditioners[k]->operator()(input, output)) +
			mPreconditioners[k]->logJacobian(input, output);
	}
}



template<class CD, class PC>
Eigen::MatrixXd CMT::PatchModel<CD, PC>::sample(int num_samples) const {
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

		int m = mOutputIndices[i].first;
		int n = mOutputIndices[i].second;

		if(mMaxPCs < 0) {
			samples.row(m * mCols + n) = mConditionalDistributions[i].sample(input);
		} else {
			if(!mPreconditioners[i])
				throw Exception("Model has to be initialized first.");
			MatrixXd inputPc = mPreconditioners[i]->operator()(input);
			MatrixXd outputPc = mConditionalDistributions[i].sample(inputPc);
			samples.row(m * mCols + n) = mPreconditioners[i]->inverse(inputPc, outputPc).second;
		}
	}

	return samples;
}



template<class CD, class PC>
int CMT::PatchModel<CD, PC>::findIndex(int i, int j) const {
	// find index corresponding to pixel (i, j)
	Tuples::const_iterator it = find(mOutputIndices.begin(), mOutputIndices.end(), make_pair(i, j));

	if(it == mOutputIndices.end())
		throw Exception("Invalid indices");

	return distance(mOutputIndices.begin(), it);
}

#endif
