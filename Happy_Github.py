''' This script was just created for fun. To record happiness of learning and creating something new.'''
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import gamma
import csv
import os


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

# compare the two evaluation utility functions' execution time
util_vecotr = np.array([0.8, 0.75, 0.9])
cumu_util = np.array([1.5, 1.3, 1.2])
range_intercept = [10, 30, 100, 300, 1000, 3000, 10000]
range_scale = [0.2, 0.3, 0.5, 0.8, 1.0]
exp_x = 3.5

for i in range(1000):
    _scale = range_scale[np.random.randint(len(range_scale))]
    res = (1 - gamma.cdf(cumu_util, a=exp_x / _scale, scale=_scale))

for i in range(1000):
    res = util_vecotr * np.exp(-0.5 * cumu_util)

# %% functionality of csv

with open(os.path.join(os.path.dirname(__file__), 'Evaluation result', 'Grid_search_instance.csv'),
          'w', newline='') as csvFile:
    fileHeader = ['index', 'a1', 'intercept', 'shape', 'scale', 'penalty', 'score']
    writer = csv.writer(csvFile)
    writer.writerow(fileHeader)

# %% add new rows
with open(os.path.join(os.path.dirname(__file__), 'Evaluation result', 'Grid_search_instance.csv'),
          'a', newline='') as csvFile:
    add_info = ["2", 150, 3, 2, 5, 0.1, 0.01]
    writer = csv.writer(csvFile)
    writer.writerow(add_info)
    # or 列表里的列表
    # writer.writerows([fileHeader, d1, d2])