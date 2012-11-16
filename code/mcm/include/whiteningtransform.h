#ifndef WHITENINGTRANSFORM_H
#define WHITENINGTRANSFORM_H

#include "lineartransform.h"

namespace MCM {
	class WhiteningTransform : public LinearTransform {
		public:
			WhiteningTransform(ArrayXXd data); 
	};
}

#endif
