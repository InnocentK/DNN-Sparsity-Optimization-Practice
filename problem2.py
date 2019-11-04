#
#
#

import numpy as np

#
def p2_2(lr=0.1, W0=[ [0.0],[0.0],[0.0] ], epochs=50):
	
	X1 = np.array([-1.0,2.0,0.0])
	y1 = -7.0
	X2 = np.array([3.0,0.0,1.0])
	y2 = 5.0
	X3 = np.array([2.0,-1.0,3.0])
	y3 = 11.0
	L1 = 0.0
	L2 = 0.0
	L3 = 0.0
	
	X = np.array([X1,X2,X3], dtype=np.float64)
	W = np.array(W0, dtype=np.float64)
	old_W = np.array(W0, dtype=np.float64)
	y = np.array([y1,y2,y3], dtype=np.float64)
	L = np.array([L1,L2,L3], dtype=np.float64)
	#grad = [0,0,0]
	all_loss = [L]
	
	for i in range(epochs):
		
		# Updating Loss
		for j in range(len(L)):
			loss = (X[j]*W[j] - y[j]) ** 2
			L[j] = sum(loss) / len(loss)
		all_loss.append(L)

		# Calculating gradients
		grad = np.array([0.0,0.0,0.0], dtype=np.float64)
		for k,_ in enumerate(W):
			grad[k] = (W[k] - old_W[k])
		avg_grad = sum(grad) / len(grad)

		# Updating weights
		old_W = np.copy(W)
		for n,weight in enumerate(W):
			if i >= 1:
				W[n][0] = weight[0] - lr * avg_grad
			else:
				W[n][0] = weight[0] - lr
	return all_loss

def main():

	loss = p2_2()
	print(loss)

	return 0
main()