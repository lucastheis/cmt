#ifndef REGULARIZER_H
#define REGULARIZER_H

#include "Eigen/Core"

namespace CMT {
	using Eigen::MatrixXd;

	class Regularizer {
		public:
			enum Norm { L1, L2 };

			Regularizer(double lambda = 0.0, Norm norm = L2);
			Regularizer(MatrixXd A, Norm norm = L2, double lambda = 1.0);

			double evaluate(const MatrixXd& parameters);
			MatrixXd gradient(const MatrixXd& parameters);

		private:
			const bool mUseMatrix;
			const Norm mNorm;
			const double mLambda;
			const MatrixXd mMatrix;
			MatrixXd mMatrixMatrix;
	};
}

#endif
