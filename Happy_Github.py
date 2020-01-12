''' This script was just created for fun. To record happiness of learning and creating something new.'''
from matplotlib import pyplot as plt
import numpy as np


def generate_y(_x, offset, steepness=1):
    y = 1 - 1 / (1 + np.exp(-steepness * (_x - offset)))
    return y


print("Let's go learning something")

# %% try new objective functions (Jan.10)

figure = plt.figure()
# 新建一个3d绘图对象


# 100 linearly spaced numbers
offset = 1.4
x = np.linspace(0, 2 * offset, 200)
y = generate_y(x, offset, steepness=5)
y1 = generate_y(x, offset, steepness=3)
y2 = generate_y(x, offset, steepness=7)

y3 = generate_y(x, offset + 0.5, steepness=5)

y0 = generate_y(x, 0.6, steepness=5)


# setting the axes at the centre
fig = plt.figure(dpi=200)
ax = fig.add_subplot(1, 1, 1)
ax.locator_params(nbins=20, axis='x')
ax.grid(True, linestyle='--', alpha=0.6)

formula = r'$f(x) = 1 - \frac{1}{1 + \exp(-(x-5)}$'

# ax.text(6, 0.8, formula,fontsize=12)

# plot the function
plt.plot(x, y, 'r', label='offset:1.4, steep:5')
plt.plot(x, y1, 'b', linestyle='--', label='offset:1.4, steep:3')
plt.plot(x, y2, 'k', linestyle='-.', label='offset:1.4, steep:7')
plt.plot(x, y3, 'cyan', linestyle='-.', label='offset:1.9, steep:5')

plt.plot(x, y0, 'orange', linestyle='-.', label='offset:0.6, steep:5')

plt.legend()

# show the plot
plt.show()
