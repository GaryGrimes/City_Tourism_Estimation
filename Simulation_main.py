# -*- coding: utf-8 -*-
"""
This script is to generate statistics for observed visit and trip frequency.
Also used to evaluate the simualtion results .
Last modified on Dec. 18
"""

import pandas as pd
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import slvr.SimDataProcessing as sim_data
from SolverUtility_ILS import SolverUtility
import multiprocessing as mp
import math
import datetime
import progressbar as pb

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class Tourist(object):
    tourist_count = 0

    def __init__(self, index, uid):
        self.index, self.uid = index, uid
        self.trip_df, self.time_budget = None, 0
        self.path_pdt, self.path_obs = None, None
        self.preference = None
        Tourist.tourist_count += 1


class Trips(object):
    trip_count = 0

    def __init__(self, uid, origin, dest):
        self.uid = uid
        self.o, self.d = origin, dest
        # self.first_trip_dummy = 0  # whether is the first trip of current tourist
        self.trip_index = None
        self.dep_time, self.arr_time = (0, 0), (0, 0)  # time formatted as tuple
        self.mode = None  # if not empty, formmated into a tuple with at most 6 candidates
        self.food_cost, self.sovn_cost = None, None


def penalty2score(*args):
    if args:
        _scores = (1 / np.array(args) * 10000) ** 20
        return _scores
    else:
        return []


def score2penalty(*args):
    if args:
        _penalty = 10000 / (np.array(args) ** (1 / 20))
        return _penalty
    else:
        return []


def change_travel_time(target: tuple) -> None:
    pass


def change_fare(target: tuple) -> None:
    pass


# split the arr into N chunks
def chunks(arr, m):
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[_:_ + n] for _ in range(0, len(arr), n)]


def eval_fun(para):
    # divide population into chunks to initiate multi-processing.
    n_cores = mp.cpu_count()
    pop = chunks(agent_database, n_cores)
    # for i in pop:
    #     print(len(i))  # 尽可能平均

    jobs = []
    penalty_queue = mp.Queue()  # queue, to save results for multi_processing

    # start process

    for idx, chunk in enumerate(pop):
        _alpha = list(para[:2])
        _beta = [5] + list(para[2:])
        data_input = {'alpha': _alpha, 'beta': _beta,
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
    return penalty_total  # unit transformed from km to m


def eval_fun_null(beta):
    # todo preference也去掉。重新定义solver.
    # divide population into chunks to initiate multi-processing.
    n_cores = mp.cpu_count()
    pop = chunks(agent_database, n_cores)
    # for i in pop:
    #     print(len(i))  # 尽可能平均

    jobs = []
    penalty_queue = mp.Queue()  # queue, to save results for multi_processing

    # start process

    for idx, chunk in enumerate(pop):
        alpha = list(beta[:2])
        beta = [5] + list(beta[2:])
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
    return penalty_total  # unit transformed from km to m


def parse_observed_trips(_agents: list):
    enmerated_agents = 0
    observed_trip_table = np.zeros((37, 37), dtype=int)

    for _agent in _agents:
        # attraction indices in solver start from 0 (in survey start from 1)
        pref, observed_path = _agent.preference, list(np.array(_agent.path_obs) - 1)

        if pref is None or observed_path is None:
            continue
        # skip empty paths (no visited location)
        if len(observed_path) < 3:
            continue
        # parse trip frequency
        for _idx in range(len(observed_path) - 1):
            _o, _d = observed_path[_idx], observed_path[_idx + 1]
            try:
                observed_trip_table[_o, _d] += 1
            except IndexError:
                continue
        enmerated_agents += 1
    return enmerated_agents, observed_trip_table


def eval_fun_trips(para):
    # divide population into chunks to initiate multi-processing.
    n_cores = mp.cpu_count()
    pop = chunks(agent_database, n_cores)
    # for i in pop:
    #     print(len(i))  # 尽可能平均

    jobs = []
    penalty_queue = mp.Queue()  # queue, to save results for multi_processing

    # start process

    for idx, chunk in enumerate(pop):
        _alpha = list(para[:2])
        _beta = [5] + list(para[2:])
        data_input = {'alpha': _alpha, 'beta': _beta,
                      'phi': phi,
                      'util_matrix': utility_matrix,
                      'time_matrix': edge_time_matrix,
                      'cost_matrix': edge_cost_matrix,
                      'dwell_matrix': dwell_vector,
                      'dist_matrix': edge_distance_matrix}

        process = mp.Process(target=SolverUtility.solver_trip_stat, args=(penalty_queue, idx, node_num, chunk),
                             kwargs=data_input, name='P{}'.format(idx + 1))
        jobs.append(process)
        process.start()

    for j in jobs:
        # wait for processes to complete and join them
        j.join()
    print('Jobs successfully joined.')
    # retrieve parameter penalties from queue
    trip_table = np.zeros((37, 37), dtype=int)
    while True:
        if penalty_queue.empty():  # 如果队列空了，就退出循环
            break
        else:
            cur_idx = penalty_queue.get()
            name = 'predicted_trip_table_{}.pickle'.format(cur_idx)
            with open(os.path.join(os.path.dirname(__file__), 'slvr', 'SimInfo', 'temp', name),
                      'rb') as file:
                trip_segment = pickle.load(file)  # note: agent = tourists here
                trip_table += trip_segment

    return trip_table  # unit transformed from km to m


def eval_fun_util_tuples(para):
    # divide population into chunks to initiate multi-processing.
    n_cores = mp.cpu_count()
    pop = chunks(agent_database, n_cores)
    # for i in pop:
    #     print(len(i))  # 尽可能平均

    jobs = []
    penalty_queue = mp.Queue()  # queue, to save results for multi_processing

    # start process

    for idx, chunk in enumerate(pop):
        _alpha = list(para[:2])
        _beta = [5] + list(para[2:])
        data_input = {'alpha': _alpha, 'beta': _beta,
                      'phi': phi,
                      'util_matrix': utility_matrix,
                      'time_matrix': edge_time_matrix,
                      'cost_matrix': edge_cost_matrix,
                      'dwell_matrix': dwell_vector,
                      'dist_matrix': edge_distance_matrix}

        process = mp.Process(target=SolverUtility.solver_util_scatter, args=(penalty_queue, idx, node_num, chunk),
                             kwargs=data_input, name='P{}'.format(idx + 1))
        jobs.append(process)
        process.start()

    for j in jobs:
        # wait for processes to complete and join them
        j.join()
    print('Jobs successfully joined.')
    # retrieve parameter penalties from queue
    res_tuples = []

    while True:
        if penalty_queue.empty():  # 如果队列空了，就退出循环
            break
        else:
            cur_idx = penalty_queue.get()
            name = 'utility_tuples_{}.pickle'.format(cur_idx)
            with open(os.path.join(os.path.dirname(__file__), 'slvr', 'SimInfo', 'temp', 'scatter plot', name),
                      'rb') as _file:
                tuple_segment = pickle.load(_file)  # note: agent = tourists here
                res_tuples.extend(tuple_segment)
                print('Enumerated res from process {}, res_tuple added {} elements. Now size: {}'.format(cur_idx, len(
                    tuple_segment), len(res_tuples)))

    return res_tuples  # unit transformed from km to m


def parse_pdt_trip(_dir):
    trip_table = np.zeros((37, 37), dtype=int)
    filenames = []
    for root, dirs, files in os.walk(_dir, topdown=False):  # os_walk只能在当前目录，不能深入！
        for name in files:
            if name.startswith('predicted_trip_table'):
                filenames.append(os.path.join(root, name))
    while filenames:
        name = filenames.pop()
        with open(name, 'rb') as _file:
            trip_segment = pickle.load(_file)  # note: agent = tourists here
            trip_table += np.array(trip_segment).reshape((37, 37))
    return trip_table


def parse_pdt_tuples(_dir):
    res_tuples = []

    filenames = []
    for root, dirs, files in os.walk(_dir, topdown=False):
        for name in files:
            if name.startswith('utility_tuples'):
                filenames.append(os.path.join(root, name))
    while filenames:
        name = filenames.pop()
        with open(name, 'rb') as _file:
            tuple_segment = pickle.load(_file)  # note: agent = tourists here
            res_tuples.extend(tuple_segment)
            print('Tuple size: {}'.format(len(tuple_segment)))
    return res_tuples


if __name__ == '__main__':
    # Data preparation
    # %% read tourist agents
    with open(os.path.join(os.path.dirname(__file__), 'slvr', 'Database', 'transit_user_database.pickle'),
              'rb') as file:
        agent_database = pickle.load(file)  # note: agent = tourists here

    print('Setting up agents...')
    print('Parsing if any agent violates outbound visit...')
    # for agents in agent database, 看observed trip里 o和d有超过47的吗？
    cnt = 0
    for _idx, _agent in enumerate(agent_database):
        error_visit = []
        for _visit in _agent.path_obs:
            if _visit > 47:
                error_visit.append(_visit)
        if error_visit:
            cnt += 1
            print('Error visits: {} for agent with index {}'.format(error_visit, _idx))
    print('Total {} error found'.format(cnt))
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

    # %% evaluation for a single set of parameter
    # numerical gradient using * parameter
    s = [0, 0, 0, 0]
    test_flag = input("1. Initiate penalty evaluation test? Any key to procceed 'Enter' to skip.")
    if test_flag:
        test_obj = eval_fun(s)
        print('Test penalty for beta* :', test_obj)
    # todo null case的话把preference也去掉, 即 node utility之和只= sum(U_ik)
    # %% statistics of interest
    enumerated_usrs, obs_trip_table = parse_observed_trips(agent_database)

    # observed trip tables 要不要scaled to the whole population's size?

    # predicted trip tables
    s_opt = [-1.286284872, -0.286449175, 0.691566901, 0.353739632]

    if input('2. Evaluate predicted trip table given current optimal set of parameters? Enter to skip.'):
        predicted_trip_tables = eval_fun_trips(s_opt)
    else:
        trip_temp_dir = os.path.join(os.path.dirname(__file__), 'slvr', 'SimInfo', 'temp')
        predicted_trip_tables = parse_pdt_trip(trip_temp_dir)

    error_trip_table = predicted_trip_tables - obs_trip_table
    error_trip_percentage = (predicted_trip_tables - obs_trip_table) / obs_trip_table
    # %% Error between the utilities of observed trip and predicted trip, for each tourist using PT
    # todo 比较两个trip table的相对偏差，+-量
    if input('3. Evaluate utility tuples of observed and predicted trips? Enter to skip, any key to proceed.'):
        to_plot_tuples = eval_fun_util_tuples(s_opt)
    else:
        tuple_temp_dir = os.path.join(os.path.dirname(__file__), 'slvr', 'SimInfo', 'temp', 'scatter plot')
        to_plot_tuples = parse_pdt_tuples(tuple_temp_dir)
