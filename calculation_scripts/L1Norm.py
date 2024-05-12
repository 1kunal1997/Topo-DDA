import numpy as np
import matplotlib.pyplot as plt


posOld = "E:\\Calculations\\2024May10\\HalfCylinder_penalty_parabolic_coeff0.0to0.5_linear\\Structure_Values\\"
posNew = "E:\\Calculations\\2024May10\\HalfCylinder8_it400_eps_0.01_penalty_parabolic_coeff0.0to0.5_linear\\Structure_Values\\"

all_errors = [0]*400

for i in range(400):
    corestructureold = np.loadtxt(posOld + "Structure" + str(i) + ".txt")
    corestructurenew = np.loadtxt(posNew + "Structure" + str(i) + ".txt")

    for j in range(len(corestructurenew)):
        all_errors[i] += abs(corestructureold[j] - corestructurenew[j])

plt.figure(1)
plt.plot(all_errors)
#plt.yscale('log')
plt.title("L1 Norm")
plt.ylabel("Norm")
plt.xlabel("Iteration")
plt.show()

