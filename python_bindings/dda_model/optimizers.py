import numpy as np

class AdamOptimizer:
    def __init__(self, beta1=0.9, beta2=0.99):
        self.beta1 = beta1
        self.beta2 = beta2
        self.V = 0
        self.S = 0
        self.timestep = 0

    def __call__(self, gradients):

        result = np.array(gradients)
        print("Using Adam Optimizer.")
        self.V = self.beta1 * self.V + (1 - self.beta1) * gradients / (1 - self.beta1**(self.timestep + 1))
        self.S = self.beta2 * self.S + (1 - self.beta2) * (np.power(gradients, 2)) / (1 - self.beta2**(self.timestep + 1))

        for i in range(len(gradients)):
            result[i] = self.V[i] / (np.sqrt(self.S[i]) + 1e-8)
        
        self.timestep += 1

        return result
    
# NOTE: Implementation does not seem to match one that Ben linked in PR (May 2024)
class AdaGradOptimizer:
    def __init__(self):
        self.squaredGradient = 0

    def __call__(self, gradients):

        result = np.array(gradients)
        print("Using Adagrad Optimizer.")
        self.squaredGradient += np.power(gradients, 2)
        result = gradients / (self.squaredGradient + 1)
        return result