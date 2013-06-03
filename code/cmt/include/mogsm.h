#ifndef CMT_MOGSM_H
#define CMT_MOGSM_H

#include "mixture.h"

namespace CMT {
	class MoGSM : public Mixture {
		public:
			MoGSM(int dim, int numComponents, int numScales = 6);
	};
}



inline CMT::MoGSM::MoGSM(int dim, int numComponents, int numScales) : Mixture(dim) {
	for(int k = 0; k < numComponents; ++k)
		addComponent(new GSM(dim, numScales));
}

#endif
