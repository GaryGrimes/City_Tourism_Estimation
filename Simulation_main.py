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


def evaluation(_beta, _itr):
    """Evaluation of each set of parameters using MultiProcessing"""
    global phi, utility_matrix, dwell_vector, edge_time_matrix, edge_cost_matrix, edge_distance_matrix

    print('------ Iteration {} ------\n'.format(_itr + 1))

    jobs = []
    penalty_queue = mp.Queue()  # queue, to save results for multi_processing

    # calculate evaluation time
    start_time = datetime.datetime.now()

    # evaluation with MultiProcessing for each parameter in current generation
    for idx, parameter in enumerate(_s):
        print('\nStarting process {} in {}'.format(idx + 1, len(_s)))
        parameter = parameter.tolist()  # convert to list
        # check existence of parameter in memory
        if parameter in memo_parameter:
            # sent back penalty tuple if exists in history
            penalty_queue.put((idx, memo_penalty[memo_parameter.index(parameter)]))
            print('\nThe {}th parameter is sent from history (with index {}), with score: {}'.format(
                idx, memo_parameter.index(parameter), memo_penalty[memo_parameter.index(parameter)]))
        else:
            ALPHA = list(parameter[:2])
            BETA = [5] + list(parameter[2:])
            data_input = {'alpha': ALPHA, 'beta': BETA,
                          'phi': phi,
                          'util_matrix': utility_matrix,
                          'time_matrix': edge_time_matrix,
                          'cost_matrix': edge_cost_matrix,
                          'dwell_matrix': dwell_vector,
                          'dist_matrix': edge_distance_matrix}

            # start process
            process = mp.Process(target=SolverUtility.solver, args=(penalty_queue, idx, node_num, agent_database),
                                 kwargs=data_input, name='P{}'.format(idx + 1))
            jobs.append(process)
            process.start()

    for _j in jobs:
        # wait for processes to complete and join them
        _j.join()

    # collect end time
    end_time = datetime.datetime.now()
    print('\n------ Evaluation time for current iteration: {}s ------\n'.format((end_time - start_time).seconds))

    # retrieve parameter penalties from queue
    para_penalties_tuples = []
    while True:
        if penalty_queue.empty():  # 如果队列空了，就退出循环
            break
        else:
            para_penalties_tuples.append(penalty_queue.get())

    para_penalties = []
    # sort the retrieved penalties so that it has a same order with the original parameter set 's'
    for _i in range(len(_s)):
        for _tuple in para_penalties_tuples:
            if _i == _tuple[0]:
                para_penalties.append(_tuple[1])  # Caution! 目前传回的tuple[1]是一个dict!!!
                break

    memo_parameter.extend(_s.tolist())
    memo_penalty.extend(para_penalties)

    PARAMETER[_itr] = _s  # save parameters of each iteration into the PARAMETER dict.

    scores = penalty2score(para_penalties)[0]  # functions returns ndarray

    # print evaluation scores
    print('Evaluation scores for iteration {}:'.format(_itr))
    for _i, _ in enumerate(scores):
        print('Parameter %d: a1: %.3f, a2: %.3f; b2: %.3f, b3: %.3f, with score: %.3e'
              % (_i + 1, _s[_i][0], _s[_i][1], _s[_i][2], _s[_i][3], _))
    return scores


if __name__ == '__main__':
    # Data preparation
    # %% Solver Setup
    # load agents
    print('Setting up agents...')
    agent_database = sim_data.agent_database

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


    # %% todo evaluation for a single set of parameter
    # todo null case的话把preference也去掉, 即 node utility之和只= sum(U_ik)
    # %% statistics of interest
    # todo 比较两个trip table的相对偏差，+-量
