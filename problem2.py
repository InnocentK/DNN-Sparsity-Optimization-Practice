#
#
#

import numpy as np
import math
import random

def print2CSV(loss, name="p2-2"):
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

# Quantizes gradients
def terngrad(grads, clipGrads=False, T=5):
	max_grad = max(np.abs(grads))
	tern = []

	if clipGrads and max_grad > T:
		max_grad = T

	for grad in grads:
		prob_s = 0
		if max_grad != 0:
			prob_s = abs(grad[0]) / max_grad
		new_grad = max_grad * isTarget(prob_s) * np.sign(grad[0])
		tern.append([new_grad])
	return tern

#
def p2(lr=0.1, W0=[ [0.0],[0.0],[0.0] ], epochs=50, isQuantize=False, isClip=False, thresh=1):
	
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
	y = np.array([y1,y2,y3], dtype=np.float64)
	L = np.array([L1,L2,L3], dtype=np.float64)
	
	grads = np.array(W0, dtype=np.float64)
	all_loss = []
	num_workers = len(L)
	
	for _ in range(epochs):
		
		# Updating Loss
		for j in range(num_workers):
			L[j] = ( np.dot(X[j],W) - y[j] ) ** 2

			# Calculating gradients
			grad = 2*X[j] * ( np.dot(X[j],W) - y[j])
			grads[j][0] = np.mean(grad, dtype=np.float64) # Change this for regulaization to sum

			# Gradient Clipping
			if isClip and not isQuantize and abs(grads[j][0]) > thresh:
				grads[j][0] = thresh * np.sign(grads[j][0])
		
		all_loss.append(np.log(L))
		
		# Quantizing the gradient
		if isQuantize:
			grads = np.array( terngrad(grads, isClip, thresh) )

		avg_grad = np.mean(grads, dtype=np.float64)

		# Updating weights
		for k,weight in enumerate(W):
				W[k][0] = weight[0] - lr * avg_grad
		#print(W, '\n')
		#print(avg_grad, '\n')
	return all_loss

def main():

	# Problem 2.2
	loss2_2 = p2()
	print2CSV(loss2_2,"p2-2")
	#"""
	# Problem 2.3
	loss2_3 = p2(isQuantize=True)
	print2CSV(loss2_3,"p2-3")

	# Problem 2.4
	clips = [1,2,5,10]
	for clip in clips:
		loss2_4 = p2(isClip=True, thresh=clip)
		print2CSV(loss2_4,"p2-4_T=" + str(clip))
	
	# Problem 2.5
	loss2_5 = p2(isQuantize=True, isClip=True, thresh=5)
	print2CSV(loss2_5,"p2-5")
	#"""
	return 0
main()