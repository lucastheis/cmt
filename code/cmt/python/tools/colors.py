"""
Tools for converting color images.
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@theis.io>'
__docformat__ = 'epytext'
__version__ = '0.1.0'

from numpy import array, dot
from numpy.linalg import inv, det

Kr = .299
Kg = .587
Kb = 1. - Kr - Kg

# color matrix
YCbCr = array([
	[Kr, -Kr / (1. - Kb) / 2.,              1. / 2.],
	[Kg, -Kg / (1. - Kb) / 2., -Kg / (1. - Kb) / 2.],
	[Kb,              1. / 2., -Kb / (1. - Kr) / 2.]])

# make volume preserving
YCbCr /= pow(det(YCbCr), 1. / YCbCr.shape[0])

def rgb2ycc(img):
	"""
	Converts an RGB image into a YCbCr encoded image.
	
	The color matrix used to convert the image is volume-preserving, that is,
	the determinant of the transformation is 1.

	@type  img: C{ndarray}
	@param img: an RGB or RGBA image

	@rtype: C{ndarray}
	@return: a YCbCr encoded image

	@see: L{ycc2rgb}
	"""

	if img.shape[2] == 4:
		img = img[:, :, :3]
	if img.shape[2] != 3:
		raise ValueError('Array is not an RGB image.')
	return dot(img.reshape(-1, 3), YCbCr).reshape(img.shape[0], img.shape[1], 3)



def ycc2rgb(img):
	"""
	Converts a YCbCr image back to RGB.

	@type  img: C{ndarray}
	@param img: a YCbCr encoded image

	@rtype: C{ndarray}
	@return: an RGB image

	@see: L{rgb2ycc}
	"""

	if img.shape[2] != 3:
		raise ValueError('Array is not a YCbCr image.')
	return dot(img.reshape(-1, 3), inv(YCbCr)).reshape(img.shape[0], img.shape[1], 3)



def rgb2gray(img):
	"""
	Converts the image to YCbCr and returns its luma component.

	@type  img: C{ndarray}
	@param img: an RGB or RGBA image

	@rtype: C{ndarray}
	@return: a grayscale image

	@see: L{rgb2ycc}
	"""

	if img.shape[2] == 4:
		img = img[:, :, :3]
	if img.shape[2] != 3:
		raise ValueError('Array is not an RGB image.')
	return rgb2ycc(img)[:, :, 0]
