#
#
#

def p2_2(lr=0.1, W0=[0,0,0], epochs=50):
	
	X1 = [-1,2,0]
	y1 = -7
	X2 = [3,0,1]
	y2 = 5
	X3 = [2,-1,3]
	y3 = 11
	L1 = 0
	L2 = 0
	L3 = 0
	
	X = [X1,X2,X3]
	W = W0
	y = [y1,y2,y3]
	L = [L1,L2,L3]
	#grad = [[0,0,0], [0,0,0], [0,0,0]]
	
	for _ in range(epochs):
		
		# Updating Loss
		for j in range(len(L)):
			L[j] = (X[j]*W[j] - y[j]) ** 2

		# Calculating gradients
		#grad = [0,0,0]
		#for k,Xi in enumerate(X):
			#grad[k] += Xi
		
		# Updating weights
		for weight in W:
			weight = weight - lr #* (DECAY**(EPOCHS // DECAY_EPOCHS)) <- see Lab 2


def main():

	return 0
main()