#ifndef CMT_BINNINGTRANSFORM_H
#define CMT_BINNINGTRANSFORM_H

#include "affinetransform.h"

namespace CMT {
	class BinningTransform : public AffineTransform {
		public:
			BinningTransform(int binning, int dimIn, int dimOut = 1);

			int binning();

		private:
			int mBinning;
	};
}



inline int CMT::BinningTransform::binning() {
	return mBinning;
}

#endif
