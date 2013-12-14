"""
Tools for generating masks.
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@theis.io>'
__docformat__ = 'epytext'
__version__ = '0.1.0'

from numpy import max, iterable, squeeze, zeros, zeros_like, any

def generate_masks(input_size, output_size=1, observed=None):
	"""
	Generates some basic input and output masks.

	If C{input_size} is an integer, the number of columns of the mask will be
	that integer. If C{input_size} is a list or tuple, a mask with multiple channels
	is created, which can be used with RGB images, for example.
	
	By default, the input region will cover the upper half of the mask, also known as a
	*causal neighborhood*. If any of the channels is observed, the input region in that
	channel will cover a full square neighborhood around the output region.

	Examples:

		>>> input_mask, output_mask = generate_masks(8, 2)
		>>> input_mask, output_mask = generate_masks([3, 7, 7], 1, [1, 0, 0])

	@type  input_size: C{int} / C{list}
	@param input_size: determines the size of the input region

	@type  output_size: C{int}
	@param output_size: determines the size of the output region

	@type  observed: C{list}
	@param observed: can be used to indicate channels which are observed

	@rtype: C{tuple}
	@return: one input mask and one output mask
	"""

	if not iterable(input_size):
		if iterable(observed):
			input_size = [input_size] * len(observed)
		else:
			input_size = [input_size]

	if observed is None:
		observed = [False] * len(input_size)

	if len(observed) != len(input_size):
		raise ValueError("Incompatible `input_size` and `observed`.")

	num_channels = len(input_size)
	num_cols = max(input_size)
	num_rows = num_cols if any(observed) else (num_cols + 1) // 2 + output_size // 2

	input_mask = zeros([num_rows, num_cols, num_channels], dtype='bool')
	output_mask = zeros_like(input_mask)

	tmp1 = (num_cols + 1) // 2
	tmp2 = output_size // 2
	tmp3 = (output_size + 1) // 2

	for k in range(num_channels):
		offset = tmp1 - (input_size[k] + 1) // 2

		if observed[k]:
			input_mask[
				offset:num_cols - offset,
				offset:num_cols - offset, k] = True
		else:
			input_mask[offset:tmp1 + tmp2, offset:num_cols - offset, k] = True

			for i in range(output_size):
				input_mask[
					tmp1 + tmp2 - i - 1,
					tmp1 - tmp3:, k] = False
				output_mask[
					tmp1 + tmp2 - i - 1,
					tmp1 - tmp3:tmp1 + output_size // 2, k] = True

	if input_mask.shape[2] == 1:
		input_mask.resize(input_mask.shape[0], input_mask.shape[1])
		output_mask.resize(output_mask.shape[0], output_mask.shape[1])

	return input_mask, output_mask
