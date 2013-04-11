#include <vector>
using std::vector;

#include "Eigen/Core"
using Eigen::ArrayXXd;
using Eigen::MatrixXd;
using Eigen::Array;
using Eigen::Dynamic;

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

	Tuples inputIndices = maskToIndices(inputMask);
	Tuples outputIndices = maskToIndices(outputMask);

	if(inputIndices.size() > 1)
		throw Exception("Only one-dimensional outputs are currently supported.");

	int rowOffset = inputIndices[0].first;
	int colOffset = inputIndices[0].second;

	// initialize conditional distributions with copy constructor
	for(int i = 0; i < rows; ++i)
		for(int j = 0; j < cols; ++j) {
			// compute input indices for
			Tuples indices;

			for(Tuples::iterator it = indices.begin(); it != indices.end(); ++it) {
				// location of input pixel in patch
				int m = i + it->first - rowOffset;
				int n = j + it->second - colOffset;

				if(m >= 0 && m < mask.rows() && n >= 0 && n < mask.cols())
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

	for(int i = 0; i < rows * cols; ++i)
		converged &= mConditionalDistributions[i].train(data, data, params);

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
