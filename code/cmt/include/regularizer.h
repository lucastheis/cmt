#ifndef CMT_REGULARIZER_H
#define CMT_REGULARIZER_H

#include "Eigen/Core"

namespace CMT {
	using Eigen::MatrixXd;

	class Regularizer {
		public:
			enum Norm { L1, L2 };

			Regularizer(double strength = 0., Norm norm = L2);
			Regularizer(MatrixXd transform, Norm norm = L2, double strength = 1.);

			double evaluate(const MatrixXd& parameters) const;
			MatrixXd gradient(const MatrixXd& parameters) const;

		private:
			bool mUseMatrix;
			Norm mNorm;
			double mStrength;
			MatrixXd mTransform;
			MatrixXd mTT;
	};
}

#endif
