#
#
#

import numpy as np
import math
import random

def print2CSV(loss, name="p2-2"):
	out_file = open("./" + name + "_results.csv", "w")

	out_file.write("Worker 1, Worker 2, Wroker 3\n")
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
		prob_s = abs(grad) / max_grad
		new_grad = max_grad * isTarget(prob_s) * np.sign(grad)
		tern.append(new_grad)
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
	old_W = np.array(W0, dtype=np.float64)
	y = np.array([y1,y2,y3], dtype=np.float64)
	L = np.array([L1,L2,L3], dtype=np.float64)
	#grad = [0,0,0]
	all_loss = [L]
	
	for i in range(epochs):
		
		# Updating Loss
		for j in range(len(L)):
			loss = (X[j]*W[j] - y[j]) ** 2
			L[j] = math.log(sum(loss) / len(loss))
		all_loss.append(L)

		# Calculating gradients
		grad = np.array([0.0,0.0,0.0], dtype=np.float64)
		for k,_ in enumerate(W):
			grad[k] = (W[k] - old_W[k]) * L[k]

			# Gradient Clipping
			if isClip and not isQuantize and abs(grad[k]) > thresh:
				grad[k] = thresh * np.sign(grad[k])

		# Quantizing the gradient
		if isQuantize:
			grad = terngrad(grad, isClip, thresh)
		avg_grad = sum(grad) / len(grad)

		# Updating weights
		old_W = np.copy(W)
		for n,weight in enumerate(W):
			if i >= 1:
				W[n][0] = weight[0] - lr * avg_grad
			else:
				W[n][0] = weight[0] - lr
		print(W)
		print()
	return all_loss

def main():

	# Problem 2.2
	loss2_2 = p2()
	print2CSV(loss2_2,"p2-2")

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
	print2CSV(loss2_5,"p2-5_T")

	return 0
main()