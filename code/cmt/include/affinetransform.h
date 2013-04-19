#ifndef AFFINETRANSFORM_H
#define AFFINETRANSFORM_H

#include "transform.h"
#include "affinepreconditioner.h"

namespace CMT {
	class AffineTransform : public AffinePreconditioner, public Transform {
		public:
			AffineTransform(const VectorXd& meanIn, const MatrixXd& preIn, int dimOut = 1);
			AffineTransform(
				const VectorXd& meanIn,
				const MatrixXd& preIn,
				const MatrixXd& preInInv,
				int dimOut = 1);

			virtual int dim() const;
			virtual int dimTr() const;

			virtual pair<ArrayXXd, ArrayXXd> operator()(const ArrayXXd& input, const ArrayXXd& output) const;
			virtual pair<ArrayXXd, ArrayXXd> inverse(const ArrayXXd& input, const ArrayXXd& output) const;

			virtual pair<ArrayXXd, ArrayXXd> adjustGradient(
				const ArrayXXd& inputGradient,
				const ArrayXXd& outputGradient) const;

		protected:
			AffineTransform();
	};
}

#endif
