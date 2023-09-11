import math
import numpy as np
import matplotlib.pyplot as plt
def kl(p, q):
	if p == 0:
		return (1-p)*math.log((1-p)/(1-q))
	elif p == 1:
		return p*math.log(p/q)
	else:
		return p*math.log(p/q) + (1-p)*math.log((1-p)/(1-q))
   
def find_q(value,p):
    low = p
    high = 1.0
    while high - low > 0.01:
        mid = (high + low)/2
        if kl(p,mid) == value:
            return mid
        elif kl(p,mid) > value:
            high = mid
        else:
            low = mid
    return (high + low)/2

# print(find_q(math.log(10)/5,0.7))

# emp_means= [0.6, 0.7]
# ucb_t = emp_means + 2*np.sqrt(math.log(time)/count)

# count = np.ones(3)
# count[0] +=  1
# print(3/count) 

task2p2s = [ 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9 ]
kl_ = []
for i in task2p2s:
      print(i+0.1,i)
      kl_.append(kl(i+0.1,i))

plt.plot(task2p2s,kl_)
plt.show()

