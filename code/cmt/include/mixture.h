#ifndef CMT_MIXTURE_H
#define CMT_MIXTURE_H

#include <vector>
#include "Eigen/Core"
#include "distribution.h"

namespace CMT {
	using Eigen::VectorXd;
	using Eigen::MatrixXd;
	using Eigen::ArrayXXd;

	using std::vector;

	class Mixture : public Distribution {
		public:
			class Component : public Distribution {
				public:
					struct Parameters {
						public:
							int maxIter;

							Parameters();
					};

					virtual Component* copy() = 0;

					virtual bool train(
						const MatrixXd& data,
						const Parameters& parameters = Parameters()) = 0;
					virtual bool train(
						const MatrixXd& data,
						const Array<double, 1, Dynamic>& weights,
						const Parameters& parameters = Parameters()) = 0;
			};

			struct Parameters {
				public:
					int maxIter;

					Parameters();
			};

			Mixture(int dim);
			virtual ~Mixture();

			virtual int dim() const;
			virtual int numComponents() const;

			virtual VectorXd priors() const;
			virtual void setPriors(const VectorXd& priors);

			virtual Component* operator[](int i);
			virtual const Component* operator[](int i) const;
			virtual void addComponent(Component* component);

			virtual MatrixXd sample(int numSamples) const;

			virtual ArrayXXd posterior(const MatrixXd& data);
			virtual Array<double, 1, Dynamic> logLikelihood(const MatrixXd& data) const;

			virtual bool train(
				const MatrixXd& data,
				const Parameters& parameters = Parameters());

		protected:
			int mDim;
			VectorXd mPriors;
			vector<Component*> mComponents;
	};
}



inline int CMT::Mixture::dim() const {
	return mDim;
}



inline Eigen::VectorXd CMT::Mixture::priors() const {
	return mPriors;
}



inline void CMT::Mixture::setPriors(const VectorXd& priors) {
	mPriors = priors;
}



inline int CMT::Mixture::numComponents() const {
	return mComponents.size();
}



inline CMT::Mixture::Component* CMT::Mixture::operator[](int i) {
	return mComponents[i];
}



inline const CMT::Mixture::Component* CMT::Mixture::operator[](int i) const {
	return mComponents[i];
}

#endif
