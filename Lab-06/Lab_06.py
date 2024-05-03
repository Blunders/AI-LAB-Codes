\\ "A not so simple"

import numpy as np
import re

def load_text_file(book):
    with open(book, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def preprocess_text(text, max_length):
    # Removing punctuations and converting to lowercase
    text = re.sub(r'[^a-zA-Z]', " ", text)
    text = " ".join(text.split()).lower()[:max_length]
    return text

def initialize_dictionary():
    dictionary = {chr(i + 97): i for i in range(26)}
    dictionary[" "] = 26
    return dictionary

def initialize_parameters(text, dictionary):
    # Initialize observation sequence
    O = np.array([dictionary[char] for char in text])
    
    # Initial state distribution
    pi = np.array([0.525483, 0.474517])
    
    # Observable sequence
    B = np.array([[0.03735, 0.03408, 0.03455, 0.03828, 0.03782, 0.03922, 0.03688, 
                   0.03408, 0.03875, 0.04062, 0.03735, 0.03968, 0.03548, 0.03735, 
                   0.04062, 0.03595, 0.03641, 0.03408, 0.04062, 0.03548, 0.03922, 
                   0.04062, 0.03455, 0.03595, 0.03408, 0.03408, 0.03688],
                  [0.03909, 0.03537, 0.03537, 0.03909, 0.03583, 0.03630, 0.04048, 
                   0.03537, 0.03816, 0.03909, 0.03490, 0.03723, 0.03537, 0.03909, 
                   0.03397, 0.03397, 0.03816, 0.03676, 0.04048, 0.03443, 0.03537, 
                   0.03955, 0.03816, 0.03723, 0.03769, 0.03955, 0.03397]])
    
    # Transition matrix
    A = np.array([[0.47468, 0.52532], [0.51656, 0.48344]])
    
    return O, pi, B, A

def alpha_pass(A, B, pi, O):
    T = len(O)
    N = len(A)
    c = np.zeros([T, 1])
    alpha = np.zeros([T, N])
    
    # Initialization
    c[0][0] = 0
    for x in range(N):
        alpha[0][x] = pi[x] * B[x][O[0]]
        c[0][0] += alpha[0][x]
    c[0][0] = 1 / c[0][0]
    for x in range(N):
        alpha[0][x] *= c[0][0]
    
    # Induction
    for t in range(1, T):
        c[t][0] = 0
        for x in range(N):
            alpha[t][x] = 0
            for y in range(N):
                alpha[t][x] += alpha[t - 1][y] * A[y][x] * B[x][O[t]]
            c[t][0] += alpha[t][x]
        c[t][0] = 1 / c[t][0]
        for x in range(N):
            alpha[t][x] *= c[t][0]
    
    return alpha, c

def beta_pass(A, B, O, c):
    T = len(O)
    N = len(A)
    beta = np.zeros([T, N])
    
    # Initialization
    for x in range(N):
        beta[T - 1][x] = c[T - 1][0]
    
    # Induction
    for t in range(T - 2, -1, -1):
        for x in range(N):
            beta[t][x] = 0
            for y in range(N):
                beta[t][x] += A[x][y] * B[y][O[t + 1]] * beta[t + 1][y]
            beta[t][x] *= c[t][0]
    
    return beta

def gamma_pass(alpha, beta, A, B, O):
    T = len(O)
    N = len(A)
    gamma = np.zeros([T, N])
    di_gamma = np.zeros([T, N, N])
    
    for t in range(T - 1):
        for x in range(N):
            gamma[t][x] = 0
            for y in range(N):
                di_gamma[t][x][y] = alpha[t][x] * A[x][y] * B[y][O[t + 1]] * beta[t + 1][y]
                gamma[t][x] += di_gamma[t][x][y]
    
    for x in range(N):
        gamma[T - 1][x] = alpha[T - 1][x]
    
    return gamma, di_gamma

def re_estimate(gamma, di_gamma, A, B, pi):
    T = len(gamma)
    N = len(A)
    M = len(B[0])
    
    # Re-estimating pi
    for x in range(N):
        pi[x] = gamma[0][x]
    
    # Re-estimating A
    for x in range(N):
        denominator = sum(gamma[t][x] for t in range(T - 1))
        for y in range(N):
            numerator = sum(di_gamma[t][x][y] for t in range(T - 1))
            A[x][y] = numerator / denominator
    
    # Re-estimating B
    for x in range(N):
        denominator = sum(gamma[t][x] for t in range(T))
        for y in range(M):
            numerator = sum(gamma[t][x] for t in range(T) if O[t] == y)
            B[x][y] = numerator / denominator
    
    return A, B, pi

def log_prob(c):
    return -np.sum(np.log(c))

if __name__ == "__main__":
    book = 'war_and_peace.txt'
    max_text_length = 100000
    
    text = load_text_file(book)
    text = preprocess_text(text, max_text_length)
    
    dictionary = initialize_dictionary()
    O, pi, B, A = initialize_parameters(text, dictionary)
    
    # Alpha pass
    alpha, c = alpha_pass(A, B, pi, O)
    
    # Beta pass
    beta = beta_pass(A, B, O, c)
    
    # Gamma pass
    gamma, di_gamma = gamma_pass(alpha, beta, A, B, O)
    
    # Re-estimate parameters
    A, B, pi = re_estimate(gamma, di_gamma, A, B, pi)
    
    # Compute log[P(O|lambda)]
    logProb = log_prob(c)
    
    print("A: \n", A)
    print("B: \n", np.concatenate((dictionary.keys(), np.round_(B, decimals=7)), axis=0).T)
    print("pi: ", np.round_(pi, decimals=7))
    print("logProb: ", logProb)



