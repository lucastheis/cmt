#ifndef PCAPRECONDITIONER_H
#define PCAPRECONDITIONER_H

#include "whiteningpreconditioner.h"

namespace CMT {
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

			inline VectorXd eigenvalues() const;

		protected:
			VectorXd mEigenvalues;
	};
}



VectorXd CMT::PCAPreconditioner::eigenvalues() const {
	return mEigenvalues;
}

#endif
