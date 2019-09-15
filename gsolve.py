"""Solve for imaging system response function.

 Given a set of pixel values observed for several pixels in several
 images with different exposure times, this function returns the
 imaging system’s response function g as well as the log film irradiance
 values for the observed pixels.

 Assumes:

 Zmin = 0
 Zmax = 255

 Arguments:

 Z(i,j) is the pixel values of pixel location number i in image j
 B(j) is the log delta t, or log shutter speed, for image j
 l is lamdba, the constant that determines the amount of smoothness
 w(z) is the weighting function value for pixel value z

 Returns:

 g(z) is the log exposure corresponding to pixel value z
 lE(i) is the log film irradiance at pixel location i
"""
import numpy as np

def gsolve(Z, B, lambda_, w, Zmin, Zmax):

	n = Zmax + 1
	num_px, num_im = Z.shape
	A = np.zeros((num_px * num_im + n, n + num_px))
	b = np.zeros((A.shape[0]))
	
	# include the data fitting equations
	k = 0
	for i in range(num_px):
		for j in range(num_im):
			wij = w[Z[i,j]]
			A[k, Z[i,j]] = wij
			A[k, n+i] = -wij
			b[k] = wij * B[j]
			k += 1

	# fix the curve by setting its middle value to 0
	A[k, n//2] = 1
	k += 1

	# include the smoothness equations
	for i in range(n-2):
		A[k, i]= lambda_ * w[i+1]
		A[k, i+1] = -2 * lambda_ * w[i+1]
		A[k, i+2] = lambda_ * w[i+1]
		k += 1

	# solve the system using LLS
	output = np.linalg.lstsq(A, b)
	x = output[0]
	g = x[:n]
	lE = x[n:]
	
	return [g, lE]