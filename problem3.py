#
#
#

import numpy as np
import math
import random

def print2CSV(loss, weights, name="p3-1"):
	out_file = open("./" + name + "_results.csv", "w")

	out_file.write("Loss, Weight 1, Weight 2, Weight 3, Weight 4, Weight 5\n")
	for l,w in zip(loss,weights):
		out_file.write(str(l) + ",")
		
		for wi in w:
		 out_file.write(str(wi[0]) + ",")
		 
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
		if weight[0] != 0:
			num_nonzeros += 1

		if num_nonzeros > thresh:
			new_weights[i][0] = 0
			num_nonzeros -= 1

	return new_weights

def proximal(weights, thresh, lambd, doTrim):
	new_weights = np.copy(weights)
	smallest_vals = None
	if doTrim:
		smallest_vals = np.argsort(new_weights.flatten())

		for i in range(3):
			idx = smallest_vals[i]
			weight = new_weights[idx]

			if abs(weight[0]) > lambd:
				new_weights[idx][0] = weight[0] - (lambd * np.sign(weight[0]) )
			else:
				new_weights[idx][0] = 0

	else:
		for i,weight in enumerate(weights):

			if abs(weight[0]) > lambd:
				new_weights[i][0] = weight[0] - (lambd * np.sign(weight[0]) )
			else:
				new_weights[i][0] = 0

	return new_weights

#
def p3(lr=0.02, W0=[ [0.0],[0.0],[0.0],[0.0],[0.0] ], epochs=200, doL0Prune=False, thresh=2, doLasso=False, lambd=0.2, doProximal=False, mu=0.004, doTrim=False, only2Pnts=False):
	
	X1 = np.array([1.0,-2.0,-1.0,-1.0,1.0])
	y1 = -7.0
	X2 = np.array([2.0,-1.0,2.0,0.0,-2.0])
	y2 = -1.0
	X3 = np.array([-1.0,0.0,2.0,2.0,1.0])
	y3 = -1.0
	
	X = np.array([X1,X2,X3], dtype=np.float64)
	W = np.array(W0, dtype=np.float64)
	y = np.array([y1,y2,y3], dtype=np.float64)
	L = 0.0
	
	if only2Pnts:
		X = np.array([X1,X2], dtype=np.float64)
		y = np.array([y1,y2], dtype=np.float64)

	grads = np.array(W0, dtype=np.float64)
	all_loss = []
	all_weights = []
	num_workers = len(X)
	
	for i in range(epochs):
		
		# Updating Loss
		losses = np.zeros(num_workers)
		for j in range(num_workers):
			losses[j] = ( np.dot(X[j],W) - y[j] ) ** 2

			# Calculating gradients
			grad = None

			# Minimizes the L1 Regularization
			if doLasso:
				grad = ( 2*X[j] * ( np.dot(X[j],W) - y[j]) ) + lambd * np.sign(W)
			else:
				grad = ( 2*X[j] * ( np.dot(X[j],W) - y[j]) )

			grads[j][0] = np.sum(grad, dtype=np.float64)
		
		L = sum(losses)
		all_loss.append(np.log(L))
		avg_grad = np.mean(grads, dtype=np.float64)

		# Updating weights
		for k,weight in enumerate(W):
				W[k][0] = weight[0] - lr * avg_grad

		# Iterative Pruning
		if doL0Prune:
			W = l0_prune(W, thresh)

		# Proximal Update
		if doProximal:
			W = proximal(W,L,mu, doTrim)

		all_weights.append(W)

	return all_loss, all_weights

def main():

	# Problem 3.1
	loss3_1, weight3_1 = p3()
	print2CSV(loss3_1, weight3_1, "p3-1")

	# Problem 3.2
	loss3_2, weight3_2 = p3(doL0Prune=True)
	print2CSV(loss3_2, weight3_2, "p3-2")

	# Problem 3.3
	lambdas = [0.2,0.5,1.0,2.0]
	for l in lambdas:
		loss3_3, weight3_3 = p3(doLasso=True, lambd=l)
		print2CSV(loss3_3, weight3_3, "p3-3_lamda=" + str(l))

	# Problem 3.4
	prox_lamdas = [0.004, 0.01, 0.02, 0.04]
	for i,mu in enumerate(prox_lamdas):
		loss3_4, weight3_4 = p3(doLasso=True, doProximal=True, lambd=lambdas[i], mu=mu)
		print2CSV(loss3_4, weight3_4, "p3-4_lamda=" + str(lambdas[i]) + "_mu=" + str(mu) )

	# Problem 3.5
	lambdas = [1.0, 2.0, 5.0, 10.0]
	prox_lamdas = [0.004, 0.01, 0.02, 0.04]
	for i,mu in enumerate(prox_lamdas):
		loss3_5, weight3_5 = p3(doLasso=True, doProximal=True, lambd=lambdas[i], mu=mu, doTrim=True)
		print2CSV(loss3_5, weight3_5, "p3-5_lamda=" + str(lambdas[i]) + "_mu=" + str(mu) )


	# Problem 3.6
	# Problem 3.6.1
	loss3_6_1, weight3_6_1 = p3(only2Pnts=True)
	print2CSV(loss3_6_1, weight3_6_1, "p3-6-1")

	# Problem 3.6.2
	loss3_6_2, weight3_6_2 = p3(doL0Prune=True, only2Pnts=True)
	print2CSV(loss3_6_2, weight3_6_2, "p3-6-2")

	# Problem 3.6.3
	lambdas = [0.2,0.5,1.0,2.0]
	for l in lambdas:
		loss3_6_3, weight3_6_3 = p3(doLasso=True, lambd=l, only2Pnts=True)
		print2CSV(loss3_6_3, weight3_6_3, "p3-6-3_lamda=" + str(l))

	# Problem 3.6.4
	prox_lamdas = [0.004, 0.01, 0.02, 0.04]
	for i,mu in enumerate(prox_lamdas):
		loss3_6_4, weight3_6_4 = p3(doLasso=True, doProximal=True, lambd=lambdas[i], mu=mu, only2Pnts=True)
		print2CSV(loss3_6_4, weight3_6_4, "p3-6-4_lamda=" + str(lambdas[i]) + "_mu=" + str(mu) )

	# Problem 3.6.5
	lambdas = [1.0, 2.0, 5.0, 10.0]
	prox_lamdas = [0.004, 0.01, 0.02, 0.04]
	for i,mu in enumerate(prox_lamdas):
		loss3_6_5, weight3_6_5 = p3(doLasso=True, doProximal=True, lambd=lambdas[i], mu=mu, doTrim=True, only2Pnts=True)
		print2CSV(loss3_6_5, weight3_6_5, "p3-6-5_lamda=" + str(lambdas[i]) + "_mu=" + str(mu) )

	return 0
main()