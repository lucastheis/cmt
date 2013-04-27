#ifndef PCAPRECONDITIONER_H
#define PCAPRECONDITIONER_H

#include "affinepreconditioner.h"

namespace CMT {
	class PCAPreconditioner : public AffinePreconditioner {
		public:
			PCAPreconditioner(
				const ArrayXXd& input,
				const ArrayXXd& output,
				double varExplained = 99.,
				int numPCs = -1);
			PCAPreconditioner(
				const VectorXd& eigenvalues,
				const VectorXd& meanIn,
				const VectorXd& meanOut,
				const MatrixXd& preIn,
				const MatrixXd& preInInv,
				const MatrixXd& preOut,
				const MatrixXd& preOutInv,
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
