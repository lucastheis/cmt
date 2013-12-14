#ifndef CMT_TRAINABLE_H
#define CMT_TRAINABLE_H

#include <utility>
#include "Eigen/Core"
#include "lbfgs.h"
#include "conditionaldistribution.h"

namespace CMT {
	using std::pair;

	using Eigen::Dynamic;
	using Eigen::Matrix;
	using Eigen::Map;
	using Eigen::MatrixXd;
	using Eigen::ArrayXXd;

	class Trainable : public ConditionalDistribution {
		public:
			class Callback {
				public:
					virtual ~Callback();
					virtual Callback* copy() = 0;
					virtual bool operator()(int iter, const Trainable& cd) = 0;
			};

			struct Parameters {
				public:
					int verbosity;
					int maxIter;
					double threshold;
					int numGrad;
					int batchSize;
					Callback* callback;
					int cbIter;
					int valIter;
					int valLookAhead;

					ArrayXXd* valInput;
					ArrayXXd* valOutput;

					Parameters();
					Parameters(const Parameters& params);
					virtual ~Parameters();
					virtual Parameters& operator=(const Parameters& params);
			};

			virtual ~Trainable();

			virtual void initialize(const MatrixXd& input, const MatrixXd& output);
			virtual void initialize(const pair<ArrayXXd, ArrayXXd>& data);

			virtual bool train(
				const MatrixXd& input,
				const MatrixXd& output,
				const Parameters& params = Parameters());
			virtual bool train(
				const MatrixXd& input,
				const MatrixXd& output,
				const MatrixXd& inputVal,
				const MatrixXd& outputVal,
				const Parameters& params = Parameters());
			virtual bool train(
				const pair<ArrayXXd, ArrayXXd>& data,
				const Parameters& params = Parameters());
			virtual bool train(
				const pair<ArrayXXd, ArrayXXd>& data,
				const pair<ArrayXXd, ArrayXXd>& dataVal,
				const Parameters& params = Parameters());

			virtual double checkGradient(
				const MatrixXd& input,
				const MatrixXd& output,
				double epsilon = 1e-5,
				const Parameters& params = Parameters());
			virtual double checkPerformance(
				const MatrixXd& input,
				const MatrixXd& output,
				int repetitions = 2,
				const Parameters& params = Parameters());

			virtual int numParameters(const Parameters& params) const = 0;
			virtual lbfgsfloatval_t* parameters(const Parameters& params) const = 0;
			virtual void setParameters(
				const lbfgsfloatval_t* x,
				const Parameters& params) = 0;

			virtual double parameterGradient(
				const MatrixXd& input,
				const MatrixXd& output,
				const lbfgsfloatval_t* x,
				lbfgsfloatval_t* g,
				const Parameters& params) const = 0;

			virtual MatrixXd fisherInformation(
				const MatrixXd& input,
				const MatrixXd& output,
				const Parameters& params = Parameters());

		protected:
			typedef Map<Matrix<lbfgsfloatval_t, Dynamic, Dynamic> > MatrixLBFGS;
			typedef Map<Matrix<lbfgsfloatval_t, Dynamic, 1> > VectorLBFGS;

			struct InstanceLBFGS {
				Trainable* cd;
				const Parameters* params;

				const MatrixXd* input;
				const MatrixXd* output;

				// used for validation error based early stopping
				const MatrixXd* inputVal;
				const MatrixXd* outputVal;
				double logLoss;
				int counter;
				lbfgsfloatval_t* parameters;
				double fx;

				InstanceLBFGS(
					Trainable* cd,
					const Trainable::Parameters* params,
					const MatrixXd* input,
					const MatrixXd* output);
				InstanceLBFGS(
					Trainable* cd,
					const Trainable::Parameters* params,
					const MatrixXd* input,
					const MatrixXd* output,
					const MatrixXd* inputVal,
					const MatrixXd* outputVal);
				~InstanceLBFGS();
			};

			static int callbackLBFGS(
				void*,
				const lbfgsfloatval_t*,
				const lbfgsfloatval_t*,
				const lbfgsfloatval_t,
				const lbfgsfloatval_t,
				const lbfgsfloatval_t,
				const lbfgsfloatval_t,
				int, int, int);

			static lbfgsfloatval_t evaluateLBFGS(
				void*,
				const lbfgsfloatval_t* x,
				lbfgsfloatval_t* g,
				int, double);

			virtual bool train(
				const MatrixXd& input,
				const MatrixXd& output,
				const MatrixXd* inputVal = 0,
				const MatrixXd* outputVal = 0,
				const Parameters& params = Parameters());
	};
}

#endif
