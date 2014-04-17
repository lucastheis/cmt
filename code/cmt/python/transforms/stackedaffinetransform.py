from cmt.transforms import AffineTransform
from numpy import vstack
from scipy.linalg import block_diag

class StackedAffineTransform(object):
	"""
	Concatenates affine transforms.
	
	This is useful if the inputs consists of several parts that need to be
	be processed separately.

	Example:

		>>> transform = StackedAffineTransform(
		>>>		PCATransform(inputs[:N], num_pcs=10),
		>>>		BinningTransform(binning=5, dim_in=inputs.shape[0] - N))

	Using this transform, the first C{N} dimensions are preprocessed using PCA, while
	the remaining dimensions are preprocessed by summing neighboring values.
	"""

	def __new__(cls, *args, **kwargs):
		return AffineTransform(
			mean_in=vstack([transform.mean_in for transform in args]),
			pre_in=block_diag(*[transform.pre_in for transform in args]),
			**kwargs)
