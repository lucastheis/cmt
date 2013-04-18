#ifndef WHITENINGPRECONDITIONER_H
#define WHITENINGPRECONDITIONER_H

#include "affinepreconditioner.h"

namespace CMT {
	class WhiteningPreconditioner : public AffinePreconditioner {
		public:
			WhiteningPreconditioner(const ArrayXXd& input, const ArrayXXd& output);
			WhiteningPreconditioner(
				const VectorXd& meanIn,
				const VectorXd& meanOut,
				const MatrixXd& preIn,
				const MatrixXd& preInInv,
				const MatrixXd& preOut,
				const MatrixXd& preOutInv,
				const MatrixXd& predictor);
	};
}

#endif
