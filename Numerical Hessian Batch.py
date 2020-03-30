"""Methods for getting numerical Hessian and pseudo t value for the estimated parameters.
Input: estimated (best) set of parameters
Output: numerical Hessian and pseudo t value
"""

import numpy as np
import numdifftools as nd
import sympy
import os
import pickle
import slvr.SimDataProcessing as sim_data
import datetime
import math
import multiprocessing as mp
from SolverUtility_ILS import SolverUtility
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Tourist(object):
    tourist_count = 0

    def __init__(self, index, uid):
        self.index, self.uid = index, uid
        self.trip_df, self.time_budget = None, 0
        self.path_pdt, self.path_obs = None, None
        self.preference = None
        Tourist.tourist_count += 1


def f(beta):
    x, y = beta[0], beta[1]
    # return 2 * x ** 2 * y + 3 * y
    return - x ** 2 - y ** 2 + 3


def f_1(beta, pos):  # *loc 输入的是i,j i.e. 在哪两个方向上变化
    _center = list(beta)
    _epsilon = 0.01 * np.array(beta)  # todo 正式的code里epsilon是变化的，按array来定
    for _ in range(len(_epsilon)):
        _epsilon[_] = 0.0001 if _epsilon[_] < 0.0001 else _epsilon[_]
    _left, _right = list(_center), list(_center)
    _left[pos], _right[pos] = _left[pos] - _epsilon[pos], _right[pos] + _epsilon[pos]

    _l_res = (f(_left) - f(beta)) / -_epsilon[pos]
    _r_res = (f(_right) - f(beta)) / _epsilon[pos]
    return _l_res, _r_res


def f_2(beta, loc):
    d_1, d_2 = loc[0], loc[1]
    _epsilon = 0.01 * np.array(beta)
    for _ in range(len(_epsilon)):
        _epsilon[_] = 0.0001 if _epsilon[_] < 0.0001 else _epsilon[_]
    _left = list(beta)  # 仅取前差分算
    _left[d_2] -= _epsilon[d_2]
    return (f_1(_left, d_1)[0] - f_1(beta, d_1)[0]) / -_epsilon[d_2]  # 取前差分算


# split the arr into N chunks
def chunks(arr, m):
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[_:_ + n] for _ in range(0, len(arr), n)]


def eval_fun(s):  # s is a single set of parameters, not species
    # divide population into chunks to initiate multi-processing.
    n_cores = mp.cpu_count()
    pop = chunks(agent_database, n_cores)
    # for i in pop:
    #     print(len(i))  # 尽可能平均

    jobs = []
    penalty_queue = mp.Queue()  # queue, to save results for multi_processing

    for idx, chunk in enumerate(pop):
        alpha = s[0]
        # 5 was replaced by a larger value of 100!!!

        beta = {'intercept': s[1], 'shape': s[2], 'scale': s[3]}
        data_input = {'alpha': alpha, 'beta': beta,
                      'phi': phi,
                      'util_matrix': utility_matrix,
                      'time_matrix': edge_time_matrix,
                      'cost_matrix': edge_cost_matrix,
                      'dwell_matrix': dwell_vector,
                      'dist_matrix': edge_distance_matrix}

        process = mp.Process(target=SolverUtility.solver, args=(penalty_queue, idx, node_num, chunk),
                             kwargs=data_input, name='P{}'.format(idx + 1))
        jobs.append(process)
        process.start()

    for j in jobs:
        # wait for processes to complete and join them
        j.join()

    # retrieve parameter penalties from queue, with total penalty for all enumerated tourists in km
    penalty_total = 0
    while True:
        if penalty_queue.empty():  # 如果队列空了，就退出循环
            break
        else:
            penalty_total += penalty_queue.get()[1]  # 0是index，1才是data
    return penalty_total


def eval_fun_null(s):  # s is a single set of parameters, not species
    # divide population into chunks to initiate multi-processing.
    n_cores = mp.cpu_count()
    pop = chunks(agent_database, n_cores)
    # for i in pop:
    #     print(len(i))  # 尽可能平均

    jobs = []
    penalty_queue = mp.Queue()  # queue, to save results for multi_processing

    for idx, chunk in enumerate(pop):
        alpha = s[0]
        # 5 was replaced by a larger value of 100!!!

        beta = {'intercept': s[1], 'shape': s[2], 'scale': s[3]}
        data_input = {'alpha': alpha, 'beta': beta,
                      'phi': phi,
                      'util_matrix': utility_matrix,
                      'time_matrix': edge_time_matrix,
                      'cost_matrix': edge_cost_matrix,
                      'dwell_matrix': dwell_vector,
                      'dist_matrix': edge_distance_matrix}

        process = mp.Process(target=SolverUtility.solver_null, args=(penalty_queue, idx, node_num, chunk),
                             kwargs=data_input, name='P{}'.format(idx + 1))
        jobs.append(process)
        process.start()

    for j in jobs:
        # wait for processes to complete and join them
        j.join()

    # retrieve parameter penalties from queue, with total penalty for all enumerated tourists in km
    penalty_total = 0
    while True:
        if penalty_queue.empty():  # 如果队列空了，就退出循环
            break
        else:
            penalty_total += penalty_queue.get()[1]  # 0是index，1才是data
    return penalty_total


def eval_fun_norm(s):
    global res_null
    # penalty in null case

    # divide population into chunks to initiate multi-processing.
    n_cores = mp.cpu_count()
    pop = chunks(agent_database, n_cores)
    # for i in pop:
    #     print(len(i))  # 尽可能平均

    jobs = []
    penalty_queue = mp.Queue()  # queue, to save results for multi_processing

    # start process

    for idx, chunk in enumerate(pop):
        alpha = s[0]
        # 5 was replaced by a larger value of 100!!!

        beta = {'intercept': s[1], 'shape': s[2], 'scale': s[3]}

        data_input = {'alpha': alpha, 'beta': beta,
                      'phi': phi,
                      'util_matrix': utility_matrix,
                      'time_matrix': edge_time_matrix,
                      'cost_matrix': edge_cost_matrix,
                      'dwell_matrix': dwell_vector,
                      'dist_matrix': edge_distance_matrix}

        process = mp.Process(target=SolverUtility.solver, args=(penalty_queue, idx, node_num, chunk),
                             kwargs=data_input, name='P{}'.format(idx + 1))
        jobs.append(process)
        process.start()

    for j in jobs:
        # wait for processes to complete and join them
        j.join()

    # retrieve parameter penalties from queue
    penalty_total = 0
    while True:
        if penalty_queue.empty():  # 如果队列空了，就退出循环
            break
        else:
            penalty_total += penalty_queue.get()[1]  # 0是index，1才是data

    pnty_null = res_null
    pnty_normed = max((pnty_null - penalty_total) / pnty_null, 0)
    return pnty_normed  # unit transformed from km to m


# %% simple example illustration
# figure = plt.figure()
# # 新建一个3d绘图对象
# ax = Axes3D(figure)
#
# # 生成x, y 的坐标集 (-2,2) 区间，间隔为 0.1
# x = np.arange(-2, 2, 0.1)
# y = np.arange(-2, 2, 0.1)
#
# # 生成网格矩阵
# X, Y = np.meshgrid(x, y)
#
# # Z 轴 函数
# Z = - np.power(X, 2) - np.power(Y, 2) + 3
#
# # 定义x,y 轴名称
# plt.xlabel("x")
# plt.ylabel("y")
#
# # 设置间隔和颜色
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="rainbow")
# # 展示
# plt.show()

res_null = None

if __name__ == '__main__':

    x, y = sympy.symbols('x y')
    print('\nFunction f(x, y) = {}\n'.format(f([x, y])))

    # ------------- Calculation ------------- #
    B_star = [0.1, 0.01]

    # gradient
    Gradient = []
    for i in range(len(B_star)):
        Gradient.append(f_1(B_star, i))

    # calculate second derivative

    SecondDerivative = []
    # second gradient
    for i in range(len(B_star)):
        temp = []
        for j in range(len(B_star)):
            temp.append(f_2(B_star, [i, j]))
        SecondDerivative.append(temp)

    Gradient = np.array(Gradient).round(3)
    SecondDerivative = np.array(SecondDerivative).round(3)

    # ---------------------------- print results  ------------------------- #
    print('\nResult at point {}:\n'.format(B_star))
    print('The gradient of f(x,y) to x: backward: {}, forward: {}\n'.format(Gradient[0][0], Gradient[0][1]))
    print('The gradient of f(x,y) to y: backward: {}, forward: {}\n'.format(Gradient[1][0], Gradient[1][1]))

    # print second derivatives
    print('The second derivative matrix:\n {}'.format(SecondDerivative))

    variance = np.linalg.inv(-SecondDerivative)
    std_err = np.sqrt(np.diag(variance))

    print('The variance matrix:\n {}'.format(variance))
    print('The std errors:\n {}'.format(std_err))

    print('t value: {}'.format(np.array(B_star) / std_err))

    # ---------------------------- print results  ------------------------- #
    print('\nNumerical Gradients and Hessian using Numdifftools: \n')

    print('The gradient of f(x,y) to x and y: {}\n'.format(nd.Gradient(f)(B_star)))
    print('The Hessian (second derivative) matrix:\n {}'.format(nd.Hessian(f)(B_star)))

    # %% simulation data processing
    #  read tourist agents
    with open(os.path.join(os.path.dirname(__file__), 'slvr', 'Database', 'transit_user_database.pickle'),
              'rb') as file:
        agent_database = pickle.load(file)  # note: agent = tourists here

    print('Setting up agents...')

    # %% setting up nodes
    node_num = sim_data.node_num  # Number of attractions. Origin and destination are excluded.

    utility_matrix = sim_data.utility_matrix
    dwell_vector = sim_data.dwell_vector

    # %% edge property
    edge_time_matrix = sim_data.edge_time_matrix

    # Edge travel cost (fare)
    edge_cost_matrix = sim_data.edge_cost_matrix

    # Edge travel distance. distance matrix for path penalty evaluation
    edge_distance_matrix = sim_data.edge_distance_matrix  # distance between attraction areas

    # %% parameter setup
    phi = sim_data.phi
    # %% Numerical Hessian using nmdf tools with beta *

    # numerical gradient using * parameter
    # s = [0.006729682, 393.7222513, 0.859129711, 0.390907255]  # Feb. 2

    # s = [0.100, 700.000, 0.367, 0.400]  # Feb. 9 ES

    s = [0.047077089, 131.3661915, 0.529121608, 2.100995628]  # Mar. 26 ES

    # current near optimal after grid search (Jan. 9)
    near_optimal = [0.027327872, 327.960607, 1, 0.4]
    NULL = [0, s[1], 0, 0]

    print('Remember to check weight in path penalty function.')

    # calculate evaluation time
    if input('Evaluate the current optimal?'):
        start_time = datetime.datetime.now()
        res = eval_fun(s)
        print('Penalty for current optimal: {}'.format(res))
        end_time = datetime.datetime.now()
        print('------ Evaluation time: {}s ------\n'.format((end_time - start_time).seconds))
        print('Evaluated penalty: %.2f' % res)

    # calculate evaluation time
    if input('Evaluate the near optimal?'):
        start_time = datetime.datetime.now()
        res = eval_fun(near_optimal)
        print('Penalty for near optimal: {}'.format(res))
        end_time = datetime.datetime.now()
        print('------ Evaluation time: {}s ------\n'.format((end_time - start_time).seconds))
        print('Evaluated penalty: %.2f' % res)

    # %% calculate evaluation time

    if input('Evaluate the null case?'):
        start_time = datetime.datetime.now()
        res_null = eval_fun_null(NULL)
        print('Penalty for null case: {}'.format(res_null))
        end_time = datetime.datetime.now()
        print('------ Evaluation time: {}s ------\n'.format((end_time - start_time).seconds))

    # %% numerical hessian and gradients
    current_optimal = s

    # res_Grad = nd.Gradient(eval_fun)(s)
    if input('Evaluate Hessian and pseudo t statistics?'):
        res_Hessian = nd.Hessian(eval_fun_norm)(current_optimal)
        print('Numerical Hessian at {}, with value: {}'.format(current_optimal, res_Hessian))

        variance = np.linalg.inv(-res_Hessian)
        std_err = np.sqrt(np.diag(variance))

        print('t value: {}'.format(np.array(current_optimal) / std_err))
