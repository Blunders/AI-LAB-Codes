import random
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, N):
        # N = number of arms
        self.N = N
        self.expRewards = [10] * N

    def actions(self):
        return list(range(0, self.N))

    def reward(self, action):
        # Generate reward with Gaussian noise
        reward = self.expRewards[action] + random.gauss(0, 0.01)
        return reward

def eGreedy(myBandit, epsilon, max_iteration):
    # Initialization
    Q = [0] * myBandit.N
    count = [0] * myBandit.N
    R_avg = [0] * max_iteration
    
    # Incremental Implementation
    for iter in range(1, max_iteration):
        if random.random() > epsilon:
            # Exploit/Greed: Choose the action with the highest estimated reward
            action = Q.index(max(Q))
        else:
            # Explore: Choose a random action
            action = random.choice(myBandit.actions())
        
        # Get reward for the chosen action
        reward = myBandit.reward(action)
        
        # Update count and estimated reward value
        count[action] += 1
        Q[action] += (reward - Q[action]) / count[action]
        
        # Update average reward
        R_avg[iter] = R_avg[iter - 1] + (reward - R_avg[iter - 1]) / iter
    
    return Q, R_avg

random.seed(10)
myBandit = Bandit(10)
Q, R_avg = eGreedy(myBandit, 0.3, 10000)

# Print actual and recovered reward values
print("Actual\tRecovered ")
for i, j in zip(myBandit.expRewards, Q):
    print(f"{i:.3f}\t{j:.3f}")

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.plot(R_avg, color = 'red')
ax1.set_title("Average rewards V/s Iteration")
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Average Reward")
ax2.plot(Q, color = 'brown')
ax2.set_title("Estimated Reward Values")
ax2.set_xlabel("Arm")
ax2.set_ylabel("Estimated Reward")
plt.show()
