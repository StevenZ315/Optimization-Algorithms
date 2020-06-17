from collections import defaultdict
from test_function import Ackley, Rastrigin, CrossIT, Sphere
import matplotlib.pyplot as plt
import numpy as np

func = Ackley(dim=2)

xlist = np.linspace(func.boundary()[0][0], func.boundary()[0][1], 100)
ylist = np.linspace(func.boundary()[1][0], func.boundary()[1][1], 100)

X, Y = np.meshgrid(xlist, ylist)
Z = np.empty(shape=X.shape)

for row in range(Z.shape[0]):
    for col in range(Z.shape[1]):
        Z[row][col] = func.function((X[row][col], Y[row][col]))

fig,ax=plt.subplots(1,1)
cp = ax.contourf(X, Y, Z, cmap='Greys', alpha=0.5)
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('Filled Contours Plot')
#ax.set_xlabel('x (cm)')
ax.set_ylabel('y (cm)')
plt.show()