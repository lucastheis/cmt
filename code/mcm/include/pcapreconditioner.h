#ifndef PCAPRECONDITIONER_H
#define PCAPRECONDITIONER_H

#include "whiteningpreconditioner.h"

namespace MCM {
	class PCAPreconditioner : public WhiteningPreconditioner {
		public:
			PCAPreconditioner(
				const ArrayXXd& input,
				const ArrayXXd& output,
				double varExplained = 99.0,
				int numPCs = -1);
			PCAPreconditioner(
				const VectorXd& eigenvalues,
				const VectorXd& meanIn,
				const VectorXd& meanOut,
				const MatrixXd& whiteIn,
				const MatrixXd& whiteInInv,
				const MatrixXd& whiteOut,
				const MatrixXd& whiteOutInv,
				const MatrixXd& predictor);

//			virtual pair<ArrayXXd, ArrayXXd> operator()(const ArrayXXd& input, const ArrayXXd& output) const;
//			virtual pair<ArrayXXd, ArrayXXd> inverse(const ArrayXXd& input, const ArrayXXd& output) const;
//
//			virtual ArrayXXd operator()(const ArrayXXd& input) const;
//			virtual ArrayXXd inverse(const ArrayXXd& input) const;

			inline VectorXd eigenvalues() const;

		protected:
			VectorXd mEigenvalues;
	};
}



VectorXd MCM::PCAPreconditioner::eigenvalues() const {
	return mEigenvalues;
}

#endif
