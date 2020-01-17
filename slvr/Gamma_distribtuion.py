"""Test new utility function adopting gamma distribution form. Last modifed on Jan.15"""
from scipy.stats import gamma
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(1, 1, dpi=150)

a = 1.99323054838
mean, var, skew, kurt = gamma.stats(a, moments='mvsk')

# plot gamma cdf. cdf(x, a, loc=0, scale=1), a>0 is shape coefficient, scale = 1.0 / lambda.
# lambda越大，scale越小，曲线越抖.
# expected value = (a-1) * scale
xmin, xmax = 0, 20
x = np.linspace(xmin, xmax, 200)
k, theta = [1, 2, 3, 5, 9], [2., 2., 2., 1.0, 0.5]
lines = []

for i in zip(k, theta):
    y = 1 - gamma.cdf(x, a=i[0], scale=i[1])
    plt.plot(x, y, label=(r'$\alpha={}, \beta={}$'.format(i[0], 1 / i[1])))

plt.xlim([xmin, xmax])
plt.ylim([0, 1])
plt.legend()
plt.show()

# 第二次plot

fig, ax = plt.subplots(1, 1, dpi=150)

xmin, xmax = 0, 5
x = np.linspace(xmin, xmax, 200)

exp_x = 3.5  # tourists perceive fatigue after visiting 3 to 4 sites. But the first several visits will not decrease so much
scale = np.array([0.1, 0.2, 0.3, 0.5, 0.7])
k, theta = exp_x / scale, scale

for i in zip(k, theta):
    y = 1 - gamma.cdf(x, a=i[0], scale=i[1])
    ax.plot(x, y, label=(r'$\alpha={}, \beta={}$'.format(i[0], 1 / i[1])))

ax.set_xlim([xmin, xmax])
ax.set_ylim([0, 1])
ax.legend()
fig.show()

# 暂时选scale为0.5, 通过修改expected value来改alpha.
print('Initially we pick scale to 0.5 and alpha to exp_x/scale = 7')
