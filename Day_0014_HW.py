import numpy as np
import matplotlib
import matplotlib.pyplot as plt
np.random.seed(1)

# x = np.random.randint(0, 50, 1000)
# y = np.random.randint(0, 50, 1000)

# corr = np.corrcoef(x, y)
# print(corr)

# plt.scatter(x,y)
# plt.show()

#負相關
x = np.random.randint(0, 50, 1000)
y = (-x) + np.random.normal(0, 10, 1000)
corr = np.corrcoef(x, y)
print(corr)
plt.scatter(x,y)
plt.show()