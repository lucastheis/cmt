#ifndef WHITENINGTRANSFORM_H
#define WHITENINGTRANSFORM_H

#include "affinetransform.h"

namespace MCM {
	class WhiteningTransform : public AffineTransform {
		public:
			WhiteningTransform(const ArrayXXd& data); 
	};
}

#endif
