#ifndef IDENTITYTRANSFORM_H
#define IDENTITYTRANSFORM_H

#include "transform.h"

namespace MCM {
	class IdentityTransform : public Transform {
		public:
			virtual int dimIn() const;
			virtual int dimOut() const;

			virtual ArrayXXd operator()(const ArrayXXd& input) const;
			virtual ArrayXXd inverse(const ArrayXXd& output) const;
	};
}

#endif
