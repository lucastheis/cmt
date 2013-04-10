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

typedef Array<bool, Dynamic, Dynamic> ArrayXXb;

template <class CD>
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
		bool train(const MatrixXd& data);

		Array<double, 1, Dynamic> logLikelihood(const MatrixXd& data) const;

		MatrixXd sample(int num_samples) const;

	protected:
		int mRows;
		int mCols;
		ArrayXXb mInputMask;
		ArrayXXb mOutputMask;
		vector<CD> mConditionalDistributions;
};



template <class CD>
PatchModel<CD>::PatchModel(
	int rows,
	int cols,
	const ArrayXXb& inputMask,
	const ArrayXXb& outputMask,
	const CD& model) : mRows(rows), mCols(cols)
{
	// this ensures that CD is a subclass of ConditionalDistribution
	ConditionalDistribution* cd = new CD(model);

	// initialize conditional distributions with copy constructor
	for(int i = 0; i < rows * cols; ++i)
		mConditionalDistributions.push_back(model);
}



template <class CD>
int PatchModel<CD>::dim() const {
	return mRows * mCols;
}



template <class CD>
CD& PatchModel<CD>::operator()(int i, int j) {
	if(j >= mCols || i >= mRows)
		throw Exception("Invalid indices.");
	return mConditionalDistributions[i * mCols + j];
};



template <class CD>
const CD& PatchModel<CD>::operator()(int i, int j) const {
	if(j >= mCols || i >= mRows)
		throw Exception("Invalid indices.");
	return mConditionalDistributions[i * mCols + j];
};



template <class CD>
bool PatchModel<CD>::initialize(const MatrixXd& data) {
	return false;
}



template <class CD>
bool PatchModel<CD>::train(const MatrixXd& data) {
	return false;
}



template <class CD>
Array<double, 1, Dynamic> PatchModel<CD>::logLikelihood(
	const MatrixXd& data) const 
{
	return Array<double, 1, Dynamic>::Zero(data.cols());
}



template<class CD>
MatrixXd PatchModel<CD>::sample(int num_samples) const {
	return MatrixXd::Zero(mRows * mCols, num_samples);
}
