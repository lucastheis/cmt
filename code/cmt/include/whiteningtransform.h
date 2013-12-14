#ifndef CMT_WHITENINGTRANSFORM_H
#define CMT_WHITENINGTRANSFORM_H

#include "affinetransform.h"

namespace CMT {
	class WhiteningTransform : public AffineTransform {
		public:
			WhiteningTransform(const ArrayXXd& input, const ArrayXXd& output);
			WhiteningTransform(const ArrayXXd& input, int dimOut = 1);
			WhiteningTransform(
				const VectorXd& meanIn,
				const MatrixXd& preIn,
				const MatrixXd& preInInv,
				int dimOut = 1);

		private:
			void initialize(const ArrayXXd& input, int dimOut);
	};
}

#endif
