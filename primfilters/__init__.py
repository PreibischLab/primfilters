import numpy as np
from skimage import feature, filters, restoration
from skimage.filters.rank import entropy
from skimage.morphology import disk
from scipy import ndimage # For 3D filters

from warnings import warn
import os

# Still need to implement: 
# For all filters dealing with a axes different pixel size

def filter_2d_or_3d(filter_fn, im, *args, **kwargs):
	"""
	This is a general function to apply 2D filters on 3D images where 3D is not implemented.
	"""
	if im.ndim==3:
		f = np.zeros(im.shape)
		for i in range(im.shape[0]):
			f[i] = filter_fn(im[i], *args, **kwargs)
	else:
		f = filter_fn(im, *args, **kwargs)
	return f


def gaussian(im, sigma=3):
	"""
	Convolution of image with a Gaussian kernel
	Suitable for N dimensions input image

	Arguments:
			im    - input image
			sigma - std of the gaussian. when sigma is a sequence it's one for each axis 
					IMPORTANT FOR DIFFERENT Z AXIS SIZE

	For classifier - recommended multiple runs with sigma 1.0-16.0
	"""
	return ndimage.gaussian_filter(im, sigma)


def sobel(im):
	"""
	Edge detector using a Sobel filter
	Input can be 3D but computes based on 2D (like in fiji WEKA)

	Arguments:
			im    - input image

	For classifier - recommended multiple runs after applying gaussian filter w sigma 1.0-16.0
	"""
	im = im.astype('int32')
	sx = ndimage.sobel(im, axis=(im.ndim-2), mode='constant')
	sy = ndimage.sobel(im, axis=(im.ndim-1), mode='constant')
	return np.hypot(sx, sy)


def prewitt(im):
	"""
	Edge detector using a Prewitt filter
	LONG RUN!

	Arguments:
			im    - input image

	For classifier - recommended multiple runs after applying gaussian filter w sigma 1.0-16.0
	"""
	im = im.astype('int32')
	sx = ndimage.prewitt(im, axis=(im.ndim-2), mode='constant')
	sy = ndimage.prewitt(im, axis=(im.ndim-1), mode='constant')
	return np.hypot(sx, sy)


def hessian(im, scale_range=(1, 5)):
	"""
	2nd order local image intensity variations - Hessian 
	CURRENTLY COMPUTES BASED ON 2D (but input can be 3D)

	Arguments:
			im    		- 	input image
			scale_range -	the sigma range of the hessian filter (step size is set to 1)

	For classifier - recommended multiple runs after applying gaussian filter w sigma 1.0-16.0
	# Importantly in WEKA there are many methods of calculating the final value of a pixel from its hessian matrix.
	# Not implemented yet.
	"""
	return filter_2d_or_3d(filters.hessian, im, scale_range)


def difference_of_gaussians(im, gaussians = (1,5)):
	"""
	Feature enhancement - DoG
	Arguments:
			im    		- 	input image
			gaussians 	- 	an array with the two gaussian values to be substructed from one another

	For classifier - typically size ratio of kernel 2 to kernel 1 is 4:1 or 5:1
					Can also approximate of the Laplacian of Gaussian - with ratio ~1.6
	"""
	return ndimage.gaussian_filter(im, gaussians[0]) - ndimage.gaussian_filter(im, gaussians[1])


def membrane_projections(im):
	"""
	Enhances membrane-like structures
	Arguments:
			im 		-	input image
	"""

	# Init initial kernel (kennel size=19 as defined in ImageJ WEKA):
	ker_size = 19
	org_k_size = ker_size*2-1
	if im.ndim==3:
		org_k = np.zeros((org_k_size, org_k_size, org_k_size))
		org_k[:,:,ker_size-1]=1
	else:
		org_k = np.zeros((org_k_size, org_k_size))
		org_k[:,ker_size-1]=1

	siz = im.shape
	siz1 = np.insert(siz, 0, 30)
	# We will get 30 images of the original image since we rotate the kernel by 6 degrees 30 times: 
	mem_proj = np.zeros(siz1)
	for i in range(30):
		# Rotate kernel:
		k = ndimage.interpolation.rotate(org_k,(i)*6, axes=(im.ndim-2,im.ndim-1), reshape=False)
		# Take only the area needed (19*19 kernel)
		if im.ndim==3:
			k = k[int(ker_size/2):int(ker_size*3/2),int(ker_size/2):int(ker_size*3/2),int(ker_size/2):int(ker_size*3/2)]
		else:
			k = k[int(ker_size/2):int(ker_size*3/2), int(ker_size/2):int(ker_size*3/2)]
		# Convolve image with kernel:
		ndimage.convolve(im, k, mode='constant', cval=0.0, output = mem_proj[i])

	# Turn 30 matrices into 1 image by sum/mean/std/median/min/max:
	mempro_sum = np.sum(mem_proj, axis=0)
	mempro_mean = np.mean(mem_proj, axis=0)
	mempro_std = np.std(mem_proj, axis=0)
	mempro_med = np.median(mem_proj, axis=0)
	mempro_min = np.amin(mem_proj, axis=0)
	mempro_max = np.amax(mem_proj, axis=0)

	#return np.stack([mempro_sum,mempro_mean,mempro_std,mempro_med,mempro_min,mempro_max], 0)
	return mempro_sum,mempro_mean,mempro_std,mempro_med,mempro_min,mempro_max


def minimum(im, win_size=5):
	"""
	The targen pixel gets the value of the minimum pixel in the neigborhood
	Arguments:
			im    	- 	input image
			win_size 	- 	neighborhood size
	"""
	return ndimage.minimum_filter(im, win_size)


def maximum(im, win_size=5):
	"""
	The targen pixel gets the value of the maximum pixel in the neigborhood
	Arguments:
			im    	- 	input image
			win_size 	- 	neighborhood size
	"""
	return ndimage.maximum_filter(im, win_size)


def median(im, win_size=5):
	"""
	The targen pixel gets the value of the median pixel in the neigborhood
	Arguments:
			im    	- 	input image
			win_size 	- 	neighborhood size
	"""
	return ndimage.median_filter(im, win_size)


def mean(im, win_size=5):
	"""
	The targen pixel gets the value of the mean of the neigborhood
	Arguments:
			im    	- 	input image
			win_size 	- 	neighborhood size
	"""
	return ndimage.uniform_filter(im, win_size)


def varience(im, win_size=5):
	"""
	The targen pixel gets the value of the varience of the neigborhood
	Arguments:
			im    	- 	input image
			win_size 	- 	neighborhood size
	"""
	mean = ndimage.uniform_filter(im, win_size)
	sqr_mean = ndimage.uniform_filter(im**2, win_size)
	return (sqr_mean - mean**2)


def anisotropic_diffusion(im, niter=20, kappa=50, gamma=0.1, step=(1.,1.), option=1):
	"""
	A diffusion (blurring) that is applied only where the gradient is small
	In fiji a1 = 0.1, 0.35, a2 = 0.9
	Arguments:
			im    - input image
			niter  - number of iterations
			kappa  - conduction coefficient 20-100. kappa controls conduction as a function of gradient.  
						If kappa is low small intensity gradients are able to block conduction and hence 
						diffusion across step edges. 
			gamma  - max value of .25 for stability. gamma controls speed of diffusion.
			step   - tuple, the distance between adjacent pixels in (y,x)
			option - 1 Perona Malik diffusion equation No 1 - favours high contrast edges over low contrast ones.
			         2 Perona Malik diffusion equation No 2 - favours wide regions over smaller ones.

	Original MATLAB code by Peter Kovesi, translated to Python and optimised by Alistair Muldal
	"""

	# initialize output array
	im = im.astype('float32')
	ann_diff = im.copy()

	if im.ndim==2:
		# initialize some internal variables
		deltaS = np.zeros_like(ann_diff)
		deltaE = deltaS.copy()
		NS = deltaS.copy()
		EW = deltaS.copy()
		gS = np.ones_like(ann_diff)
		gE = gS.copy()

		for ii in range(niter):

			# calculate the diffs
			deltaS[:-1,: ] = np.diff(ann_diff,axis=0)
			deltaE[: ,:-1] = np.diff(ann_diff,axis=1)

			# conduction gradients (only need to compute one per dim!)
			if option == 1:
			    gS = np.exp(-(deltaS/kappa)**2.)/step[0]
			    gE = np.exp(-(deltaE/kappa)**2.)/step[1]
			elif option == 2:
			    gS = 1./(1.+(deltaS/kappa)**2.)/step[0]
			    gE = 1./(1.+(deltaE/kappa)**2.)/step[1]

			# update matrices
			E = gE*deltaE
			S = gS*deltaS

			# subtract a copy that has been shifted 'North/West' by one
			# pixel. don't as questions. just do it. trust me.
			NS[:] = S
			EW[:] = E
			NS[1:,:] -= S[:-1,:]
			EW[:,1:] -= E[:,:-1]

			# update the image
			ann_diff += gamma*(NS+EW)
	else:
		step=(1.,1.,1.)
	# initialize some internal variables
		deltaS = np.zeros_like(ann_diff)
		deltaE = deltaS.copy()
		deltaD = deltaS.copy()
		NS = deltaS.copy()
		EW = deltaS.copy()
		UD = deltaS.copy()
		gS = np.ones_like(ann_diff)
		gE = gS.copy()
		gD = gS.copy()

		for ii in range(niter):

			# calculate the diffs
			deltaD[:-1,: ,:  ] = np.diff(ann_diff,axis=0)
			deltaS[:  ,:-1,: ] = np.diff(ann_diff,axis=1)
			deltaE[:  ,: ,:-1] = np.diff(ann_diff,axis=2)

			# conduction gradients (only need to compute one per dim!)
			if option == 1:
				gD = np.exp(-(deltaD/kappa)**2.)/step[0]
				gS = np.exp(-(deltaS/kappa)**2.)/step[1]
				gE = np.exp(-(deltaE/kappa)**2.)/step[2]
			elif option == 2:
				gD = 1./(1.+(deltaD/kappa)**2.)/step[0]
				gS = 1./(1.+(deltaS/kappa)**2.)/step[1]
				gE = 1./(1.+(deltaE/kappa)**2.)/step[2]

			# update matrices
			D = gD*deltaD
			E = gE*deltaE
			S = gS*deltaS

			# subtract a copy that has been shifted 'Up/North/West' by one
			# pixel. don't as questions. just do it. trust me.
			UD[:] = D
			NS[:] = S
			EW[:] = E
			UD[1:,: ,: ] -= D[:-1,:  ,:  ]
			NS[: ,1:,: ] -= S[:  ,:-1,:  ]
			EW[: ,: ,1:] -= E[:  ,:  ,:-1]

			# update the image
			ann_diff += gamma*(UD+NS+EW)

	return ann_diff


def bilateral(im, win_size=5):
	"""
	Bilateral denoising filter
	Fiji WEKA combines spatial radius of 5 and 10, with a range radius of 50 and 100
	Arguments:
		im - input image
	"""
	return filter_2d_or_3d(restoration.denoise_bilateral, im, win_size, multichannel=False)


def kuwahara(im, win_size=3):
	"""
	Smoothing filter that preserves the edges
	CURRENTLY COMPUTES BASED ON 2D (but input can be 3D)
	LONG RUN!
	Arguments:
		im 		- input image
		win_size 	- the number of pixels taken from each side to create a neighborhood.
	"""
	# Need to see if zeros is a good default of initialization - maybe nan?
	kuwah = np.zeros(im.shape)
	siz = im.shape
	rad=win_size
	if im.ndim==2:
		for j in range(siz[0]):
			for k in range(siz[1]):
				if j-rad>=0 and k-rad>=0 and j+rad<=siz[0] and k+rad<=siz[1]:
					neig_arr = im[j-rad:j+rad,k-rad:k+rad]

					std_indx = np.argmin([np.std(neig_arr[0:rad+1,0:rad+1]), np.std(neig_arr[rad:rad*2+1,0:rad+1]),
						np.std(neig_arr[0:rad+1,rad:rad*2+1]), np.std(neig_arr[rad:rad*2+1,rad:rad*2+1])])

					kuwah[j,k] = [np.mean(neig_arr[0:rad+1,0:rad+1]), np.mean(neig_arr[rad:rad*2,0:rad+1]),
						np.mean(neig_arr[0:rad+1,rad:rad*2]), np.mean(neig_arr[rad:rad*2,rad:rad*2])][std_indx]
	else:
		for i in range(siz[0]):
			for j in range(siz[1]):
				for k in range(siz[2]):
					if j-rad>=0 and k-rad>=0 and j+rad<=siz[0] and k+rad<=siz[1]:
						neig_arr = im[i,j-rad:j+rad,k-rad:k+rad]

						std_indx = np.argmin([np.std(neig_arr[0:rad+1,0:rad+1]), np.std(neig_arr[rad:rad*2+1,0:rad+1]),
							np.std(neig_arr[0:rad+1,rad:rad*2+1]), np.std(neig_arr[rad:rad*2+1,rad:rad*2+1])])

						kuwah[i,j,k] = [np.mean(neig_arr[0:rad+1,0:rad+1]), np.mean(neig_arr[rad:rad*2,0:rad+1]),
							np.mean(neig_arr[0:rad+1,rad:rad*2]), np.mean(neig_arr[rad:rad*2,rad:rad*2])][std_indx]
	return kuwah


def gabor(im, frequency=0.1, theta=0):
	"""
	Texture filter - finds specific frequency content in a specific direction
	CURRENTLY COMPUTES BASED ON 2D (but input can be 3D)
	Arguments:
		im 			- input image
		frequency 	- the number of pixels taken from each side to create a neighborhood.
		theta		- orientation in radians (angel of the gabor patch). Use values like: 0, np.pi/x
	"""
	if im.ndim==3:
		f = np.zeros(im.shape)
		f_temp = np.zeros(im.shape)
		for i in range(im.shape[0]):
			f[i], f_temp = filters.gabor(im[i], frequency, theta)
	else:
		f, f_temp = filters.gabor(im, frequency, theta)
	return f
#	gabo_real, gabo_imag = filter_2d_or_3d(filters.gabor, im, frequency, theta)
#	return gabo_real


def laplace(im):
	"""
	Blob detector using the Laplacian of of gaussian (MAYBE FIJI WEKA USES ANOTHER LAPLACE - CHECK)
	Arguments:
		im 		- input image
	RETURNED IMAGE LOOKS LIKE AN OPERATOR WAS PERFORMED ONLY IN ONE DIRECTION - NEED TO CHECK
	"""
	return ndimage.gaussian_laplace(im, sigma)


def frangi(im, scale_range=(1,10), scale_step=2):
	"""
	Detects continuous edges with the eighnvalues of the hessian - Frangi Filter
	Similar to the structure filter in fiji weka
	Arguments:
		im 				- input image
		scale_range		- range of sigmas used
		scale_step	 	- step size of the sigmas
	"""
	return filter_2d_or_3d(filters.frangi,im, scale_range, scale_step)


def entrop(im, radius=10):
	"""
	Calculates the entropy of the neighborhood for each pixel
	CURRENTLY COMPUTES BASED ON 2D (but input can be 3D)
	Arguments:
		im 				- input image
		radius			- neighborhood size (uses a disk shape)
	"""
	return filter_2d_or_3d(entropy, im, disk(radius))


# Still missing filters:
# Lipschitz filter, derivatives, structure - similar to implemented frangi, edge (just canny on 3D).


"""
used ndimage for all of those..
def stats_filter(im, radius=3):
	
	Computed for each pixel the neighborhood mean/varience/median/nim/max
	Similar code can be used for median, min, max, but those are already implemented in ndimage
	Arguments:
			im 			-	input image
			radius		-	the number of pixels taken from each side to create a neighborhood.
	
	# Need to see if zeros is a good default of initialization - maybe nan?
	mean = np.zeros(im.shape)
	var = np.zeros(im.shape)

	if im.ndim==3:
		for i in range(siz[0]):
			for j in range(siz[1]):
				for k in range(siz[2]):	
					if i-rad>=0 and j-rad>=0 and k-rad>=0 and i+rad<=siz[0] and j+rad<=siz[1] and k+rad<=siz[2]:
						neig_arr = im[i-rad:i+rad,j-rad:j+rad,k-rad:k+rad]
						mean[i,j,k] = np.mean(neig_arr)
						var[i,j,k] = np.var(neig_arr)
	else:
		for i in range(siz[0]):
			for j in range(siz[1]):
				if i-rad>=0 and j-rad>=0 and i+rad<=siz[0] and j+rad<=siz[1]:
					neig_arr = im[i-rad:i+rad,j-rad:j+rad]
					mean[i,j] = np.mean(neig_arr)
					var[i,j] = np.var(neig_arr)

	return mean, var
"""

"""
def normalize(im):
	#Image normalization (without histogram equalization)

	lmin = float(im.min())
	lmax = float(im.max())
	return np.floor((im-lmin)/(lmax-lmin)*255.)
"""