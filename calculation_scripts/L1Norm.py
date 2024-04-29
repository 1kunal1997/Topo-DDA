import numpy as np
import matplotlib.pyplot as plt

posOld = "E:\\Calculations\\2024April27\\HourglassFromTestSuite\\CoreStructure\\"
posNew = "E:\\Calculations\\2024April27\\HourglassUsingWrapper2\\CoreStructure\\"

all_errors = [0]*400

for i in range(400):
    corestructureold = np.loadtxt(posOld + "CoreStructure" + str(i) + ".txt")
    corestructurenew = np.loadtxt(posNew + "CoreStructure" + str(i) + ".txt")

    for j in range(len(corestructurenew)):
        all_errors[i] += abs(corestructureold[j] - corestructurenew[j])

plt.figure(1)
plt.plot(all_errors)
#plt.yscale('log')
plt.title("L1 Norm")
plt.ylabel("Norm")
plt.xlabel("Iteration")
plt.show()