#ifndef CMT_AFFINETRANSFORM_H
#define CMT_AFFINETRANSFORM_H

#include "affinepreconditioner.h"

namespace CMT {
	class AffineTransform : public AffinePreconditioner {
		public:
			AffineTransform(const VectorXd& meanIn, const MatrixXd& preIn, int dimOut = 1);
			AffineTransform(
				const VectorXd& meanIn,
				const MatrixXd& preIn,
				const MatrixXd& preInInv,
				int dimOut = 1);
			AffineTransform(const AffineTransform& transform);

			using AffinePreconditioner::operator();
			using AffinePreconditioner::inverse;

			virtual pair<ArrayXXd, ArrayXXd> operator()(
				const ArrayXXd& input,
				const ArrayXXd& output) const;
			virtual pair<ArrayXXd, ArrayXXd> inverse(
				const ArrayXXd& input,
				const ArrayXXd& output) const;

			virtual pair<ArrayXXd, ArrayXXd> adjustGradient(
				const ArrayXXd& inputGradient,
				const ArrayXXd& outputGradient) const;

		protected:
			AffineTransform();
	};
}

#endif
