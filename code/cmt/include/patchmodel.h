#include <vector>
using std::vector;

#include "Eigen/Core"
using Eigen::ArrayXXd;
using Eigen::MatrixXd;
using Eigen::Array;
using Eigen::Dynamic;

#include <iostream>

#include "distribution.h"
#include "exception.h"
#include "conditionaldistribution.h"
#include "tools.h"

typedef Array<bool, Dynamic, Dynamic> ArrayXXb;

template <class CD, class Parameters>
class PatchModel : public Distribution {
	public:
		PatchModel(
			int rows,
			int cols,
			const ArrayXXb& inputMask,
			const ArrayXXb& outputMask,
			const CD& model);

		int dim() const;

		CD& operator()(int i, int j);
		const CD& operator()(int i, int j) const;

		bool initialize(const MatrixXd& data);
		bool train(const MatrixXd& data, const Parameters& params);

		Array<double, 1, Dynamic> logLikelihood(const MatrixXd& data) const;

		MatrixXd sample(int num_samples) const;

	protected:
		int mRows;
		int mCols;
		ArrayXXb mInputMask;
		ArrayXXb mOutputMask;
		vector<Tuples> mInputIndices;
		vector<CD> mConditionalDistributions;
};



template <class CD, class Parameters>
PatchModel<CD, Parameters>::PatchModel(
	int rows,
	int cols,
	const ArrayXXb& inputMask,
	const ArrayXXb& outputMask,
	const CD& model) : mRows(rows), mCols(cols)
{
	// this ensures that CD is a subclass of ConditionalDistribution
	ConditionalDistribution* cd = new CD(model);
	delete cd;

	// compute locations of active pixels
	pair<Tuples, Tuples> inOutIndices = masksToIndices(inputMask, outputMask);
	Tuples& inputIndices = inOutIndices.first;
	Tuples& outputIndices = inOutIndices.second;

	if(outputIndices.size() > 1)
		throw Exception("Only one-dimensional outputs are currently supported.");

	if(inputIndices.size() != model.dimIn() || outputIndices.size() != model.dimOut())
		throw Exception("Model and masks are incompatible.");

	int rowOffset = outputIndices[0].first;
	int colOffset = outputIndices[0].second;

	// initialize conditional distributions with copy constructor
	for(int i = 0; i < rows; ++i)
		for(int j = 0; j < cols; ++j) {
			// compute input indices for
			Tuples indices;

			for(Tuples::iterator it = inputIndices.begin(); it != inputIndices.end(); ++it) {
				// location of input pixel in patch
				int m = i + it->first - rowOffset;
				int n = j + it->second - colOffset;

				if(m >= 0 && m < mRows && n >= 0 && n < mCols)
					indices.push_back(make_pair(m, n));
			}

			mInputIndices.push_back(indices);

			if(indices.size() == inputIndices.size())
				mConditionalDistributions.push_back(model);
			else
				mConditionalDistributions.push_back(CD(indices.size(), model));
		}
}



template <class CD, class Parameters>
int PatchModel<CD, Parameters>::dim() const {
	return mRows * mCols;
}



template <class CD, class Parameters>
CD& PatchModel<CD, Parameters>::operator()(int i, int j) {
	if(j >= mCols || i >= mRows)
		throw Exception("Invalid indices.");
	return mConditionalDistributions[i * mCols + j];
};



template <class CD, class Parameters>
const CD& PatchModel<CD, Parameters>::operator()(int i, int j) const {
	if(j >= mCols || i >= mRows)
		throw Exception("Invalid indices.");
	return mConditionalDistributions[i * mCols + j];
};



template <class CD, class Parameters>
bool PatchModel<CD, Parameters>::initialize(const MatrixXd& data) {
	return false;
}



template <class CD, class Parameters>
bool PatchModel<CD, Parameters>::train(const MatrixXd& data, const Parameters& params) {
	bool converged = true;

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

		converged &= mConditionalDistributions[i].train(input, output, params);
	}

	return converged;
}



template <class CD, class Parameters>
Array<double, 1, Dynamic> PatchModel<CD, Parameters>::logLikelihood(
	const MatrixXd& data) const 
{
	return Array<double, 1, Dynamic>::Zero(data.cols());
}



template<class CD, class Parameters>
MatrixXd PatchModel<CD, Parameters>::sample(int num_samples) const {
	return MatrixXd::Zero(mRows * mCols, num_samples);
}
