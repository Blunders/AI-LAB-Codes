import numpy as np


# Simplified Equation without using weights
def energy_function(neurons):
    E1 = 0
    E2 = 0
    # Loop over rows
    for i in range(8):
        acc = 0
        # Loop over columns
        for j in range(8):
            acc += neurons[i][j]
        # Accumulate energy based on the sum of neuron activations in each row
        E1 += (acc - 1) ** 2
           
    # Loop over columns
    for i in range(8):
        acc = 0
        # Loop over rows
        for j in range(8):
            acc += neurons[j][i]
        # Accumulate energy based on the sum of neuron activations in each column
        E2 += (acc - 1) ** 2


    return E1 + E2  


# %%
# Equation with weights included
def energy_function_2(neurons, weights):
    E1 = 0
    # Loop over neurons
    for i in range(64):
        for j in range(64):
            # Calculate the energy contribution of each neuron pair weighted by corresponding weight
            x1 = neurons[i // 8][j % 8]
            x2 = neurons[j // 8][i % 8]
            E1 += x1 * x2 * weights[j][i]
   
    E1 *= -0.5
   
    # Add penalty term for each neuron being activated
    for i in range(8):
        for j in range(8):
            E1 += -1 * neurons[i][j]


    return E1


# %%
# Function to print the neuron activations
def printNeurons(neurons):
    for i in range(8):
        for j in range(8):
            if(neurons[i][j] == -1):
                print(0, end = " ")  # Print 0 if neuron activation is -1
            else:
                print(neurons[i][j], end = " ")  # Print neuron activation
        print()


# %%
# Main function
def main():
    neurons = [[-1 for i in range(8)] for j in range(8)]  # Initialize neurons with -1 (inactive)
   
    # Set the first column of neurons to be activated (value 1)
    for i in range(8):
        neurons[i][0] = 1


    weights = [[0 for i in range(64)] for j in range(64)]  # Initialize weights matrix
   
    # Assign weights to connections between neurons
    for i in range(64): 
        for j in range(64):
            if i != j:
                if i % 8 == j % 8:
                    weights[i][j] = -2  # Assign -2 if neurons are in the same column
                elif i // 8 == j // 8:
                    weights[i][j] = -2  # Assign -2 if neurons are in the same row


    max_iter = 1000  # Maximum number of iterations
    E = energy_function_2(neurons, weights)  # Initial energy
    ones = []
    zeros = []
    for i in range(8):
        for j in range(8):
            if neurons[i][j] == 1:
                ones.append((i, j))  # Store indices of activated neurons
            else:
                zeros.append((i, j))  # Store indices of inactive neurons
   
    # Iterate to minimize energy
    for i in range(max_iter):
        # Randomly choose two neurons, one with value 1 and one with value 0
        x1, y1 = ones[np.random.randint(0, len(ones))]
        x2, y2 = zeros[np.random.randint(0, len(zeros))]
        neurons[x1][y1], neurons[x2][y2] = neurons[x2][y2], neurons[x1][y1]  # Swap activations
        E_new = energy_function_2(neurons, weights)  # Compute new energy
        if E_new < E:
            E = E_new  # Update energy if it's reduced
            ones = []
            zeros = []
            for i in range(8):
                for j in range(8):
                    if neurons[i][j] == 1:
                        ones.append((i, j))  # Update activated neurons
                    else:
                        zeros.append((i, j))  # Update inactive neurons
        else:
            neurons[x1][y1], neurons[x2][y2] = neurons[x2][y2], neurons[x1][y1]  # Revert the swap


    printNeurons(neurons)  # Print final neuron activations
   


# %%
main()
