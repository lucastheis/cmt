import Image
from numpy import array, asarray, squeeze, abs, min, max, percentile

def imread(filename):
	"""
	@rtype: ndarray
	@return: the image
	"""

	return array(Image.open(filename))



def imwrite(filename, img, vmin=None, vmax=None):
	"""
	A convenient wrapper for saving images using the PIL package.

	@type  filename: string
	@param filename: the place where the image shall be stored

	@type  img: array_like
	@param img: a gray-scale or RGB image

	@type  vmin: float/int
	@param vmin: can be used to manually specify which value is mapped to 0

	@type  vmax: float/int
	@param vmax: can be used to manually specify which value is mapped to 255
	"""

	Image.fromarray(imformat(img, vmin=vmin, vmax=vmax)).save(filename)



def imformat(img, symmetric=False, perc=100, vmin=None, vmax=None):
	"""
	Rescales and converts images to uint8.

	@type  img: array_like
	@param img: any image

	@type  symmetric: boolean
	@param symmetric: if true, 0. will be mapped to 128

	@type  perc: int
	@param perc: can be used to clip intensity values

	@type  vmin: float/int
	@param vmin: can be used to manually specify which value is mapped to 0

	@type  vmax: float/int
	@param vmax: can be used to manually specify which value is mapped to 255

	@rtype: ndarray
	@return: the converted image
	"""

	img = asarray(img)

	if vmin is not None and vmax is not None and vmax <= vmin:
		raise ValueError("`vmin` should be smaller than `vmax`.")

	if 'float' in str(img.dtype) \
		or max(img) > 255 \
		or min(img) < 0 \
		or vmin is not None \
		or vmax is not None:

		# rescale
		if symmetric:
			if vmax is not None:
				if vmin is not None:
					a = max([vmin, vmax])
				else:
					a = vmax
			elif vmin is not None:
				a = -vmin

			a = float(percentile(abs(img), perc))
			img = (img + a) / (2. * a) * 256.

		else:
			a, b = float(percentile(img, 100 - perc)), float(percentile(img, perc))
			if vmin is not None:
				a = vmin
			if vmax is not None:
				b = vmax
			img = (img - a) / float(b - a) * 256.

	img[img < 0] = 0
	img[img > 255] = 255

	return asarray(img, 'uint8')
