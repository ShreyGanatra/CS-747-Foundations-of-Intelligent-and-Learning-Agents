"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the base Algorithm class that all algorithms should inherit
from. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)

We have implemented the epsilon-greedy algorithm for you. You can use it as a
reference for implementing your own algorithms.
"""

import numpy as np
import math
# Hint: math.log is much faster than np.log for scalars

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# Example implementation of Epsilon Greedy algorithm
class Eps_Greedy(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # Extra member variables to keep track of the state
        self.eps = 0.1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
    
    def give_pull(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)
    
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value

# START EDITING HERE
# You can use this space to define any helper functions that you need
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
# END EDITING HERE

class UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # START EDITING HERE
        self.time=0
        self.emp_means = np.zeros(num_arms)
        self.count = np.zeros(num_arms)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        for i in range(self.num_arms):
            if self.count[i] == 0:
                return i
        ucb_t = self.emp_means + np.sqrt((2*math.log(self.time))/self.count)
        return np.argmax(ucb_t)
        # END EDITING HERE  
        
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.count[arm_index] += 1
        self.time +=1
        n = self.count[arm_index]
        value = self.emp_means[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.emp_means[arm_index] = new_value
        # END EDITING HERE


class KL_UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.time=0
        self.emp_means = np.zeros(num_arms)
        self.count = np.zeros(num_arms)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        for i in range(self.num_arms):
            if self.count[i] == 0:
                return i
        kl_ucb_t = np.zeros(self.num_arms)
        c=0
        for i in range(self.num_arms):
             value = (math.log(self.time) + c*math.log(math.log(self.time)))/self.count[i]
             kl_ucb_t[i] = find_q(value,self.emp_means[i])
        return np.argmax(kl_ucb_t)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.count[arm_index] += 1
        self.time +=1
        n = self.count[arm_index]
        value = self.emp_means[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.emp_means[arm_index] = new_value
        # END EDITING HERE

class Thompson_Sampling(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.success = np.zeros(num_arms)
        self.failure = np.zeros(num_arms)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        x_t = np.array([np.random.beta(self.success[i]+1,self.failure[i]+1) for i in range(self.num_arms)])
        return np.argmax(x_t)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        if reward:
            self.success[arm_index] +=1
        else:
            self.failure[arm_index] +=1

        # END EDITING HERE
