"""Test new utility function adopting gamma distribution form. Last modifed on Jan.15"""
from scipy.stats import gamma
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import warnings


def func(x, a, b, c):  # Logistic B equation from zunzun.com
    return a / (1.0 + np.power(x / b, c))


fig, axes = plt.subplots(1, 1, dpi=150)

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

# %% 第二次plot

fig, axes = plt.subplots(3, 1, dpi=150, figsize=(8, 18))

xmin, xmax = 0, 5
x = np.linspace(xmin, xmax, 500)

exp_x = 3.5  # tourists perceive fatigue after visiting 3 to 4 sites. But the first several visits will not decrease so much
scale = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 1.])
k, theta = exp_x / scale, scale

for i in zip(k, theta):
    # gamma distribution
    y = 1 - gamma.cdf(x, a=i[0], scale=i[1])
    # polynomial fit
    func_fit = np.polyfit(x, y, 15)
    p1 = np.poly1d(func_fit)
    y_fit = p1(x)
    # logistic fit
    # these are the same as the scipy defaults
    initialParameters = np.array([1.0, 1.0, 1.0])
    # curve fit the test data, ignoring warning due to initial parameter estimates
    warnings.filterwarnings("ignore")
    fittedParameters, pcov = curve_fit(func, x, y, initialParameters)
    y_fit_logit = func(x, *fittedParameters)

    axes[0].plot(x, y, label=(r'$\alpha={}, \beta={}$'.format(i[0], 1 / i[1])))
    axes[1].plot(x, y_fit, label=(r'$\alpha={}, \beta={}$'.format(i[0], 1 / i[1])))
    axes[2].plot(x, y_fit_logit, label=(r'$\alpha={}, \beta={}$'.format(i[0], 1 / i[1])))

for _ax in axes:
    _ax.set_xlim([xmin, xmax])
    _ax.set_ylim([0, 1])
    _ax.legend()

axes[0].set_title('Normal gamma distribution')
axes[1].set_title('Ploy fit')
axes[2].set_title('Logit fit')
fig.show()

# 暂时选scale为0.5, 通过修改expected value来改alpha.
print('Initially we pick scale to 0.5 and alpha to exp_x/scale = 7')

# %% timeit
util_array = np.array([0.8, 0.7, 0.9])
pref = np.array([0.5, 0.3, 0.3])  # just for test
node_util = np.array([0.3, 0.5, 0.7])
alpha = -50
beta = {'intercept': 3000, 'shape': 7, 'scale': 0.5}

test_func_0 = np.dot(pref, node_util * np.exp(-2 * util_array))

test_func_gamma = np.dot(pref, node_util * (1 - gamma.cdf(util_array, a=i[0], scale=i[1])))

test_func_ploy = np.dot(pref, node_util * p1(util_array))

test_func_logit = np.dot(pref, node_util * func(util_array, *fittedParameters))

# %% Reference: approximate the gamma cdf with a polynomial function using np.polyfit
# #
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 定义x、y散点坐标
#
# # 用3次多项式拟合
# f1 = np.polyfit(x, y, 3)
# print('f1 is :\n', f1)
#
# p1 = np.poly1d(f1)
# print('p1 is :\n', p1)
#
# # 也可使用yvals=np.polyval(f1, x)
# yvals = p1(x)  # 拟合y值
# print('yvals is :\n', yvals)
# # 绘图
# plot1 = plt.plot(x, y, 's', label='original values')
# plot2 = plt.plot(x, yvals, 'r', label='polyfit values')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend(loc=4)  # 指定legend的位置右下角
# plt.title('polyfitting')
# plt.show()

# %%

# https://stackoverflow.com/questions/54784383/logistic-like-curve-fitting-using-machine-learning
