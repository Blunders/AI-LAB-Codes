import random
import matplotlib.pyplot as plt

# Set random seed for reproducibility
random.seed(10)

# Binary Bandit Class
class BinaryBandit(object):
    def __init__(self):
        # Number of arms
        self.N = 2
    
    # Return available actions (arms)
    def actions(self):
        result = []
        for i in range(0, self.N):
            result.append(i)
        return result
    
    # Reward function for action on arm 1
    def reward1(self, action):
        p = [0.1, 0.2]  # Probabilities of success for each arm
        rand = random.random()
        if rand < p[action]:
            return 1  # Success
        else:
            return 0  # Failure
    
    # Reward function for action on arm 2
    def reward2(self, action):
        p = [0.8, 0.9]  # Probabilities of success for each arm
        rand = random.random()
        if rand < p[action]:
            return 1  # Success
        else:
            return 0  # Failure

# eGreedy_binary Function
def eGreedy_binary(myBandit, epsilon, max_iteration):
    # Initialization
    Q = [0] * myBandit.N  # Initialize Q-values for each action
    count = [0] * myBandit.N  # Initialize counts of each action
    R = []  # Record of rewards obtained in each iteration
    R_avg = [0] * 1  # Initialize list to track average rewards
    max_iter = max_iteration  # Maximum number of iterations

    # Incremental Implementation
    for iter in range(1, max_iter):
        if random.random() > epsilon:
            action = Q.index(max(Q))  # Exploit/ Greed
        else:
            action = random.choice(myBandit.actions())  # Explore
        r = myBandit.reward2(action)  # Obtain reward for selected action
        R.append(r)  # Record reward
        count[action] = count[action] + 1  # Update count for selected action
        # Update Q-value using incremental update rule
        Q[action] = Q[action] + (r - Q[action]) / count[action]
        # Update average reward
        R_avg.append(R_avg[iter - 1] + (r - R_avg[iter - 1]) / iter)

    return Q, R_avg, R

# Instantiate BinaryBandit object
myBandit = BinaryBandit()

# Run eGreedy_binary function
Q, R_avg, R = eGreedy_binary(myBandit, 0.2, 2000)

# Visualization
# Plot average rewards vs. Iteration
plt.figure(figsize=(8, 5))
plt.plot(R_avg, color ='red')
plt.title("Plot relating Average rewards and Iteration")
plt.xlabel("Iteration")
plt.ylabel("Average Reward")
plt.show()

# Plot Reward per iteration
plt.figure(figsize=(8, 5))
plt.plot(R, color = 'green')
plt.title("Rewards per iteration")
plt.xlabel("Iteration")
plt.ylabel("Reward")
plt.show()
