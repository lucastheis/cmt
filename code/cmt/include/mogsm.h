#ifndef CMT_MOGSM_H
#define CMT_MOGSM_H

#include "mixture.h"
#include "gsm.h"

namespace CMT {
	class MoGSM : public Mixture {
		public:
			MoGSM(int dim, int numComponents, int numScales = 6);

			int numScales() const;

		private:
			int mNumScales;
	};
}



inline CMT::MoGSM::MoGSM(int dim, int numComponents, int numScales) :
	Mixture(dim),
	mNumScales(numScales) 
{
	for(int k = 0; k < numComponents; ++k)
		addComponent(new GSM(dim, numScales));
}



inline int CMT::MoGSM::numScales() const {
	return mNumScales;
}

#endif
