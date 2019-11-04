#
#
#

import numpy as np
import math
import random

def print2CSV(loss, name="p3-1"):
	out_file = open("./" + name + "_results.csv", "w")

	out_file.write("Worker 1, Worker 2, Worker 3\n")
	for l in loss:
		for Li in l:
			out_file.write(str(Li) + ",")
		out_file.write("\n")		

	out_file.close()

# Decides if within probability
def isTarget(target_prob):

    prob = random.randint(0, 100) / 100

    if target_prob <= prob:
        return 1
    return 0

# Prunes (sets to zero) weights greater than a predefined threshold
def l0_prune(weights, thresh):
	new_weights = np.copy(weights)

	num_nonzeros = 0
	for i,weight in enumerate(weights):
		for j,Wi in enumerate(weight):

			if Wi != 0:
				num_nonzeros += 1

			if num_nonzeros > thresh:
				new_weights[i][j] = 0
				num_nonzeros -= 1

	return new_weights

# Minimizes the L1 Regularization
def lasso(weights, loss, alpha):
	new_weights = np.copy(weights)

	#lasso_reg = loss + alpha * np.sum(weights)
	min_value = -loss / alpha
	for i,weight in enumerate(weights):
		for j,Wi in enumerate(weight):
			if np.sum(new_weights) > min_value:

				new_weights[i][j] = np.sign(Wi)

	return new_weights
#
def p3(lr=0.02, W0=[ [0.0],[0.0],[0.0],[0.0],[0.0] ], epochs=200, doL0Prune=False, thresh=2, doLasso=False, lambd=0.2):
	
	X1 = np.array([1.0,-2.0,-1.0,-1.0,1.0])
	y1 = -7.0
	X2 = np.array([2.0,-1.0,2.0,0.0,-2.0])
	y2 = -1.0
	X3 = np.array([-1.0,0.0,2.0,2.0,1.0])
	y3 = -1.0
	
	X = np.array([X1,X2,X3], dtype=np.float64)
	W = np.array(W0, dtype=np.float64)
	old_W = np.array(W0, dtype=np.float64)
	y = np.array([y1,y2,y3], dtype=np.float64)
	L = 0.0
	#grad = [0,0,0]
	all_loss = []
	
	for i in range(epochs):
		
		# Updating Loss
		losses = np.zeros(len(X))
		for j in range(len(losses)):
			loss = ( X[j] * W - y[j]) ** 2
			losses[j] = np.mean(loss, dtype=np.float64)
		L = sum(losses)
		all_loss.append(math.log(L))

		# Calculating gradients
		grad = np.array([0.0,0.0,0.0,0.0,0.0], dtype=np.float64)
		for k,_ in enumerate(W):
			grad[k] = (W[k] - old_W[k]) #* L

			# Gradient Clipping
			#if isClip and not isQuantize and abs(grad[k]) > thresh:
			#grad[k] = thresh * np.sign(grad[k])

		# Updating weights
		old_W = np.copy(W)
		for n,weight in enumerate(W):
			if i >= 1:
				W[n][0] = weight[0] - lr * grad[n]#avg_grad
			else:
				W[n][0] = weight[0] - lr

		# Iterative Pruning
		if doL0Prune:
			W = l0_prune(W, thresh)

		# L1 Regularization
		if doLasso:
			W = lasso(W,L,lambd)

	return all_loss

def main():

	# Problem 3.1
	loss3_1 = p3()
	print2CSV(loss3_1,"p3-1")

	# Problem 3.2
	loss3_2 = p3(doL0Prune=True)
	print2CSV(loss3_2,"p3-2")

	# Problem 3.3
	lambdas = [0.2,0.5,1.0,2.0]
	for l in lambdas:
		loss3_3 = p3(lambd=l)
		print2CSV(loss3_2,"p3-3_lamda=" + str(l))

	return 0
main()