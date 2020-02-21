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
from matplotlib import pyplot as plt
import slvr.SimDataProcessing as sim_data
from SolverUtility_ILS import SolverUtility
import multiprocessing as mp
import math
from matplotlib import rcParams
import datetime
import matplotlib.ticker
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import Sankey as sk

plt.rcParams['font.sans-serif'] = ['Times']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic',
                               'Noto Sans CJK JP']


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


class StrategyEdge(object):
    strategy_cnt, strategy_idx = 0, 1

    def __init__(self, s_type, target: tuple, effect):
        self.idx = StrategyEdge.strategy_idx
        self.suffix = '_' + str(self.idx)
        self.type, self.target = s_type, target
        self.effect = effect
        StrategyEdge.strategy_cnt += 1
        StrategyEdge.strategy_idx += 1


class StrategyNode(object):
    strategy_cnt, strategy_idx = 0, 1

    def __init__(self, s_type, strategy: dict):
        self.idx = StrategyNode.strategy_idx
        self.suffix = '_node_' + str(self.idx)
        self.type, self.strategy = s_type, strategy

        StrategyNode.strategy_cnt += 1
        StrategyNode.strategy_idx += 1


def walklevel(some_dir, level=1):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]


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


def plot_util_in_scatter(_tuple_list, margin_rate=0.05):
    """Input: a list of tuples consisting observed and predicted path utilities. Can adjust image margin size."""
    x, y = [], []
    for _ in _tuple_list:
        x.append(_[1])  # predicted utility
        y.append((_[0]))  # observed utility

    plt.figure(dpi=400)
    x_range, y_range = max(x) - min(x), max(y) - min(y)
    x_lim, y_lim = (min(x) - margin_rate * x_range, max(x) + margin_rate * x_range), (
        min(y) - margin_rate * y_range, max(y) + margin_rate * y_range)

    plot_lim_left, plot_lim_right = min(x_lim[0], y_lim[0]), max(x_lim[1], y_lim[1])
    # plt.axis('equal')
    # plt.axis([plot_lim_left, plot_lim_right, plot_lim_left, plot_lim_right])

    plt.xlim(plot_lim_left, plot_lim_right)
    plt.ylim(plot_lim_left, plot_lim_right)

    plt.scatter(x, y, color='blue', alpha=0.5)
    # draw indentity line
    line = np.linspace(plot_lim_left, plot_lim_right, 100)
    zero_line = np.linspace(0, 0, 100)
    plt.plot(line, line, 'k--')  # identity line
    plt.plot(zero_line, line, 'r:')  # identity line
    # label and title
    plt.title('Relationship between Predicted and Observed Path Utilities')
    plt.xlabel('Predicted')
    plt.ylabel('Observed')

    # # https://stackoverflow.com/questions/17990845/how-to-equalize-the-scales-of-x-axis-and-y-axis-in-python-matplotlib
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.draw()

    plt.show()


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
        _alpha = para[0]
        _beta = {'intercept': para[1], 'shape': para[2], 'scale': para[3]}
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


def eval_fun_null(para):
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
        _alpha = para[0]
        _beta = {'intercept': para[1], 'shape': para[2], 'scale': para[3]}

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


def eval_fun_trips(para, _dir='', _suf=''):
    """Predict trip statistics given model parameters and network properties. Work with TDM strategies as well."""

    # divide population into chunks to initiate multi-processing.
    n_cores = mp.cpu_count()
    pop = chunks(agent_database, n_cores)
    # for i in pop:
    #     print(len(i))  # 尽可能平均

    jobs = []
    penalty_queue = mp.Queue()  # queue, to save results for multi_processing

    # start process

    for idx, chunk in enumerate(pop):
        _alpha = para[0]
        _beta = {'intercept': para[1], 'shape': para[2], 'scale': para[3]}

        data_input = {'alpha': _alpha, 'beta': _beta,
                      'phi': phi,
                      'util_matrix': utility_matrix,
                      'time_matrix': edge_time_matrix,
                      'cost_matrix': edge_cost_matrix,
                      'dwell_matrix': dwell_vector,
                      'dist_matrix': edge_distance_matrix}
        # todo: send _dir and _suf into process, to dump trip matrix and travel patterns
        process = mp.Process(target=SolverUtility.solver_trip_stat,
                             args=(penalty_queue, idx, node_num, chunk, _dir, _suf),
                             kwargs=data_input, name='P{}'.format(idx + 1))
        jobs.append(process)
        process.start()

    for j in jobs:
        # wait for processes to complete and join them
        j.join()
    print('Jobs successfully joined.')
    # retrieve parameter penalties from queue


def eval_fun_trips_null(para, _dir='', _suf=''):
    """Predict trip statistics given model parameters and network properties. Work with TDM strategies as well."""

    # divide population into chunks to initiate multi-processing.
    n_cores = mp.cpu_count()
    pop = chunks(agent_database, n_cores)
    # for i in pop:
    #     print(len(i))  # 尽可能平均

    jobs = []
    penalty_queue = mp.Queue()  # queue, to save results for multi_processing

    # start process

    for idx, chunk in enumerate(pop):
        _alpha = para[0]
        _beta = {'intercept': para[1], 'shape': para[2], 'scale': para[3]}

        data_input = {'alpha': _alpha, 'beta': _beta,
                      'phi': phi,
                      'util_matrix': utility_matrix,
                      'time_matrix': edge_time_matrix,
                      'cost_matrix': edge_cost_matrix,
                      'dwell_matrix': dwell_vector,
                      'dist_matrix': edge_distance_matrix}
        # todo: send _dir and _suf into process, to dump trip matrix and travel patterns
        process = mp.Process(target=SolverUtility.solver_trip_stat_null,
                             args=(penalty_queue, idx, node_num, chunk, _dir, _suf),
                             kwargs=data_input, name='P{}'.format(idx + 1))
        jobs.append(process)
        process.start()

    for j in jobs:
        # wait for processes to complete and join them
        j.join()
    print('Jobs successfully joined.')
    # retrieve parameter penalties from queue


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


def parse_trip_matrix(_dir, _suf):
    trip_table = np.zeros((37, 37), dtype=int)
    filenames = []
    for root, dirs, files in walklevel(_dir, level=0):  # os_walk只能在当前目录，不能深入！
        for name in files:
            if name.startswith('predicted_trip_table' + _suf + '_'):
                filenames.append(os.path.join(root, name))

    while filenames:
        name = filenames.pop()
        with open(name, 'rb') as _file:
            trip_segment = pickle.load(_file)  # note: agent = tourists here
            trip_table += np.array(trip_segment).reshape((37, 37))
    return trip_table


def parse_trip_pattern(_dir, _suf):
    from collections import Counter

    filenames = []
    for root, dirs, files in walklevel(_dir, level=0):  # os_walk只能在当前目录，不能深入！
        for name in files:
            if name.startswith('trip_chain' + _suf + '_'):
                filenames.append(os.path.join(root, name))

    trip_chain_table = []
    while filenames:
        name = filenames.pop()
        with open(name, 'rb') as _file:
            list_temp = pickle.load(_file)
            trip_chain_table.extend(list_temp)  # note: agent = tourists here

    # todo: indices + 1, also output Japanese one
    trip_list_withOD = ['-'.join(str(_go + 1) for _go in _path) for _path in trip_chain_table]
    trip_list_jp_withOD = ['-'.join(place_jp[_go] for _go in _path) for _path in trip_chain_table]

    trip_list_noOD = ['-'.join(str(_go + 1) for _go in _path[1:-1]) for _path in trip_chain_table]
    trip_list_jp_noOD = ['-'.join(place_jp[_go] for _go in _path[1:-1]) for _path in trip_chain_table]

    #
    trip_patterns_withod = dict(
        Counter(trip_list_withOD)
    )
    trip_patterns_jp_withod = dict(
        Counter(trip_list_jp_withOD)
    )

    df_withOD_tmp_1 = pd.DataFrame(list(trip_patterns_withod.items()))
    df_withOD_tmp_2 = pd.DataFrame(list(trip_patterns_jp_withod.items()))
    df_withOD = pd.concat([df_withOD_tmp_1, df_withOD_tmp_2], axis=1)
    df_withOD.columns = [0, 1, 2, 3]
    df_withOD = df_withOD.drop(columns=[1])
    df_withOD.columns = ['trip pattern', 'pattern in JP', 'cnt']
    df_withOD = df_withOD.sort_values(by=['cnt'], ascending=False)

    #
    trip_patterns_nood = dict(
        Counter(trip_list_noOD)
    )
    trip_patterns_jp_nood = dict(
        Counter(trip_list_jp_noOD)
    )

    df_noOD_tmp_1 = pd.DataFrame(list(trip_patterns_nood.items()))
    df_noOD_tmp_2 = pd.DataFrame(list(trip_patterns_jp_nood.items()))
    df_noOD = pd.concat([df_noOD_tmp_1, df_noOD_tmp_2], axis=1)
    df_noOD.columns = [0, 1, 2, 3]
    df_noOD = df_noOD.drop(columns=[1])
    df_noOD.columns = ['trip pattern', 'pattern in JP', 'cnt']
    df_noOD = df_noOD.sort_values(by=['cnt'], ascending=False)

    return {'raw': trip_chain_table, 'withOD': df_withOD, "no_OD": df_noOD
            }  # a dict


def parse_node_visit(_chains: list):
    """parse the number of visits to each destination from trip chains of each tourist. O, D's that are within 37
    areas are included for summation. Input: a list of all trip chains with indices starting from 0. """
    visit_frequency = np.zeros(node_num)

    for _ in _chains:
        for visit in _:
            try:
                visit_frequency[visit] += 1
            except IndexError:
                pass
    return visit_frequency


def compare_node_visit(_obs_trip_chains, _pdt_trip_chains):
    # grouped bar plot see:
    # https://matplotlib.org/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py

    node_visits_before = parse_node_visit(_obs_trip_chains)

    visits_sorted_before, idx_sorted = np.sort(node_visits_before)[::-1], np.argsort(node_visits_before)[::-1]
    place_labels_jp = [place_jp[_] for _ in idx_sorted]

    # todo: when get the node visits after strategies, need to retrieve values according to the sorted indices: idx_sorted
    node_visits_after = parse_node_visit(_pdt_trip_chains)
    visits_sorted_after = [node_visits_after[_] for _ in idx_sorted]

    x = np.arange(len(place_labels_jp))  # the label locations
    width = 0.35  # the width of the bars

    fig_wide = 9
    fig, ax = plt.subplots(figsize=(fig_wide, fig_wide / (math.sqrt(2) * 1.25)), dpi=250)

    rects1 = ax.bar(x - width / 2, visits_sorted_before, width, label='Observed', alpha=0.7)
    rects2 = ax.bar(x + width / 2, visits_sorted_after, width, label='Predicted', alpha=0.7)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Visit frequency')
    # ax.set_title('Visit frequency of places (before and after)')
    ax.set_xticks(x)
    ax.set_xlim(-1, 37)
    ax.set_xticklabels(place_labels_jp, rotation=75, ha='right')
    ax.legend()

    # ax.set_yscale('log')
    # # ax.set_yticks([0, 5, 10, 20, 50, 100, 200, 500, 1000])
    # ax.set_yticks([1, 5, 20, 50, 100, 200, 500, 1000])
    # ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(0, 1800)
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    # autolabel(rects1)  # display values on bars
    # autolabel(rects2)
    # plt.xticks(rotation=75)

    plt.grid(True, which='both', axis='y', linestyle="-.", linewidth=0.5)
    plt.tight_layout()
    plt.show()

    pass


def compare_region_visit(_obs_trip_chains, _pdt_trip_chains):
    # grouped bar plot see:
    # https://matplotlib.org/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py

    node_visits_before = parse_node_visit(_obs_trip_chains)

    region_visits_before = np.zeros(len(region_names))
    for idx, _ in enumerate(node_visits_before):
        region_visits_before[area_region_dict[idx]] += _

    visits_sorted_before, idx_sorted = np.sort(region_visits_before)[::-1], np.argsort(region_visits_before)[::-1]
    place_labels_jp = [region_names[_] for _ in idx_sorted]

    # todo: when get the node visits after strategies, need to retrieve values according to the sorted indices: idx_sorted
    node_visits_after = parse_node_visit(_pdt_trip_chains)
    region_visits_after = np.zeros(len(region_names))
    for idx, _ in enumerate(node_visits_after):
        region_visits_after[area_region_dict[idx]] += _

    visits_sorted_after = [region_visits_after[_] for _ in idx_sorted]
    error = np.abs(visits_sorted_before - visits_sorted_after)
    print('total visit frequency: observed {}, predicted {}, mean error{}'.format(
        sum(visits_sorted_before), sum(visits_sorted_after), np.average(np.average(error))))

    x = np.arange(len(place_labels_jp))  # the label locations
    width = 0.35  # the width of the bars

    fig_wide = 9
    # fig, ax = plt.subplots(figsize=(fig_wide, fig_wide / (math.sqrt(2) * 1.25)), dpi=250)
    fig, ax = plt.subplots(dpi=200)

    rects1 = ax.bar(x - width / 2, visits_sorted_before, width, label='Observed', alpha=0.7)
    rects2 = ax.bar(x + width / 2, visits_sorted_after, width, label='Predicted', alpha=0.7)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Visit frequency')
    # ax.set_title('Visit frequency of places (before and after)')
    ax.set_xticks(x)
    ax.set_xlim(-1, len(place_labels_jp))
    ax.set_xticklabels(place_labels_jp, rotation=75, ha='right')
    ax.legend()
    ax.set_ylim(0, 1800)

    # ax.set_yscale('log')
    # # ax.set_yticks([0, 5, 10, 20, 50, 100, 200, 500, 1000])
    # ax.set_yticks([1, 5, 20, 50, 100, 200, 500, 1000])
    # ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    # autolabel(rects1)  # display values on bars
    # autolabel(rects2)
    # plt.xticks(rotation=75)

    plt.grid(True, which='both', axis='y', linestyle="-.", linewidth=0.5)
    plt.tight_layout()
    plt.show()

    pass


def plot_bars(data, y_label, x_ticks):
    x = np.arange(len(data))  # the label locations

    width = 0.35  # the width of the bars
    fig_wide = 9
    fig, ax = plt.subplots(figsize=(fig_wide, fig_wide / 1.5), dpi=200)

    rects1 = ax.bar(x, data, width)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            try:
                ax.annotate('{:d}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
            except ValueError:
                ax.annotate('{:.1f}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

    autolabel(rects1)  # display values on bars

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(y_label)
    # ax.set_title('Visit frequency of places (before and after)')
    ax.set_xticks(x)
    ax.set_xlim(-1, 37)
    ax.set_xticklabels(x_ticks)

    plt.xticks(rotation=75)

    plt.grid(True, which='both', axis='y', linestyle="-.", linewidth=0.5)
    plt.show()


def plot_bars_log(data, y_label, x_ticks):
    x = np.arange(len(data))  # the label locations

    width = 0.35  # the width of the bars
    fig_wide = 9
    fig, ax = plt.subplots(figsize=(fig_wide, fig_wide / 1.5), dpi=200)

    rects1 = ax.bar(x, data, width)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            try:
                ax.annotate('{:d}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
            except ValueError:
                ax.annotate('{:.1f}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

    autolabel(rects1)  # display values on bars

    # ax.set_yscale('symlog')
    # y_max = max(data)
    # y_bound = 100 * (int(y_max / 100) + 1)
    # # ax.set_yticks([-100, -50, -20, -5, 5, 20, 50, 100, 200, y_bound])
    # ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(y_label)
    # ax.set_title('Visit frequency of places (before and after)')
    ax.set_xticks(x)
    ax.set_xlim(-1, 37)
    ax.set_xticklabels(x_ticks)

    plt.xticks(rotation=75)
    plt.yscale('symlog', linscaley=2.0)
    # plt.yscale('symlog')
    plt.grid(True, which='both', axis='y', linestyle="-.", linewidth=0.5)
    plt.show()


def plot_visit_freq_effect(_original_trip_chains, *tdm_strategy_res):
    """ Create tdm strategy plots, including the effect of strategies by grouped bar plot as well as a log-scaled
    total predicted visit frequency plot. # plot_node_effect(predicted_trip_chains['raw'], *tdm_edge_res[:2])

    """
    # plt.style.use('seaborn-colorblind')

    predicted_visit_freq = parse_node_visit(_original_trip_chains)
    # define index orders
    visits_sorted, idx_sorted = np.sort(predicted_visit_freq)[::-1], np.argsort(predicted_visit_freq)[::-1]

    place_labels_sorted = [place_jp[_] for _ in idx_sorted]

    # parse strategy results
    tdm_trip_matrices, stg_names = [], []
    data_to_plot = []
    for _ in tdm_strategy_res:
        stg_names.append('strategy' + _['stg_no'][1:])  # discard '_'
        tdm_trip_matrices.append(_['trip_matrix'])

        visit_freq_change = parse_node_visit(_['trip_chains']['raw']) - predicted_visit_freq
        data_to_plot.append([visit_freq_change[_] for _ in idx_sorted])

    # todo: visit_frequency_change in sorted order (w.r.t predicted frequency, descending)
    x = np.arange(len(place_labels_sorted))  # the label locations

    bar_cnt = len(tdm_strategy_res)

    width = (1 - 0.2) / bar_cnt  # the width of the bars

    fig_wide = 10
    fig, ax1 = plt.subplots(figsize=(fig_wide, fig_wide / (math.sqrt(2) * 1.25)), dpi=400)
    # plt.xticks(rotation=75)

    rects = []
    color_range = ['#96ceb4', '#ffcc5c', '#ff6f69']
    for _ in range(bar_cnt):
        i, n = _ + 1, bar_cnt
        loc = x + (i - 1 / 2 * (n + 1)) * width
        # rects.append(ax1.bar(loc, data_to_plot[_], width, label=stg_names[_], alpha=0.6))

        rects.append(ax1.bar(loc, data_to_plot[_], width, label=stg_names[_], color=color_range[_], alpha=0.9))

    # todo: plot grouped absolute change
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax1.set_ylabel('Change in Visit Frequency')
    ax1.set_xticks(x)
    ax1.set_xlim(-1, len(place_labels_sorted))
    ax1.set_xticklabels(place_labels_sorted, rotation=75, ha='right')

    ax1.legend()

    # For the minor ticks, use no labels; default NullFormatter.
    ax1.yaxis.set_minor_locator(MultipleLocator(5))

    ax1.grid(which='major', axis='y', linestyle="-.", linewidth=0.5)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax1.annotate('{}'.format(height),
                         xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom')

    # for _ in rects:
    #     autolabel(_)  # display values on bars

    # log scale total visit frequency on ax2
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    rect_2 = ax2.plot(x, visits_sorted, linestyle='-.', marker='d', color='tab:purple', alpha=0.3)

    ax2.set_yscale('log')
    # ax.set_yticks([0, 5, 10, 20, 50, 100, 200, 500, 1000])
    ax2.set_yticks([1, 5, 10, 20, 50, 100, 200, 500, 1000])
    ax2.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.set_xticklabels(place_labels_sorted, rotation=75, ha='right')

    ax2.set_ylabel('Predicted visit frequency (log scale)')  # , color=color)  # we already handled the x-label with ax1
    plt.tight_layout()
    plt.show()


def cal_edge_effect(_original_trips, _after_trips):
    visit_num_change = _after_trips - _original_trips
    visit_per_change = visit_num_change / _original_trips * 100
    return visit_num_change, visit_per_change


def run_time_strategy(target_areas, effect, dir, suf, run=False):
    import itertools

    global utility_matrix, edge_time_matrix, edge_cost_matrix

    stg_type = 'time'
    # initialization
    """               restore the values to original  
                      'util_matrix': utility_matrix,
                      'time_matrix': edge_time_matrix,
                      'cost_matrix': edge_cost_matrix,
                      'dwell_matrix': dwell_vector,
                      'dist_matrix': edge_distance_matrix}  """

    # should re-instantiate such that sim_data will not be dirty

    utility_matrix = np.array(sim_data.utility_matrix)
    edge_time_matrix = np.array(sim_data.edge_time_matrix)
    edge_cost_matrix = np.array(sim_data.edge_cost_matrix)

    # find all combinations of pairing ods that are sent as targets

    od_pairs_to_modify = list(itertools.combinations(target_areas, 2))

    print('Strategy #{} of Travel Time: '.format(suf))
    for _index, _ in enumerate(od_pairs_to_modify):
        print('Target {}: {} to {}'.format(_index + 1, place_jp[_[0]], place_jp[_[1]]))
        print('travel_time_before is {:.3f} min(s)'.format(edge_time_matrix[_]))
        # apply strategies (both back and forth)
        edge_time_matrix[_[::-1]] = edge_time_matrix[_] = \
            edge_time_matrix[_] * (1 - effect)

    # update time matrix
    no_update, itr = 0, 0
    # print('Starting travel_time_check...')
    for _ in range(3):
        while not no_update:
            # print('Current iteration: {}'.format(itr + 1))
            no_update = 1
            for i in range(edge_time_matrix.shape[0] - 1):
                for j in range(i + 1, edge_time_matrix.shape[0]):
                    time = edge_time_matrix[i, j]
                    shortest_node, shortest_time = 0, time
                    for k in range(edge_time_matrix.shape[0]):
                        if edge_time_matrix[i, k] + edge_time_matrix[k, j] < shortest_time:
                            shortest_node, shortest_time = k, edge_time_matrix[i, k] + edge_time_matrix[k, j]
                    if shortest_time < time:
                        no_update = 0
                        # print('travel time error between {0} and {1}, \
                        # shortest path is {0}-{2}-{1}'.format(i, j, shortest_node))
                        edge_time_matrix[j, i] = edge_time_matrix[i, j] = shortest_time
            itr += 1
            if no_update:
                # print('Travel time update complete.\n')
                pass

    for _index, _ in enumerate(od_pairs_to_modify):
        print('Target {}: travel_time_after is {:.3f} min(s)\n'.format(_index, edge_time_matrix[_]))

    # if run
    if run:
        eval_fun_trips(s_opt, _dir=dir, _suf=suf)

    predicted_trip_matrix = parse_trip_matrix(dir, suf)  # suffix is ''. no suffix
    predicted_trip_chains = parse_trip_pattern(dir, suf)

    return {'stg_no': suf, 'trip_matrix': predicted_trip_matrix, 'trip_chains': predicted_trip_chains}


def run_node_strategy(strategies: dict, dir, suf, run=False):
    global utility_matrix, edge_time_matrix, edge_cost_matrix

    """ {20: effect, 21: effect} """
    stg_type = 'util'

    utility_matrix = np.array(sim_data.utility_matrix)
    edge_time_matrix = np.array(sim_data.edge_time_matrix)
    edge_cost_matrix = np.array(sim_data.edge_cost_matrix)

    # find all combinations of pairing ods that are sent as targets

    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    print('Strategy #{} of Node utility: '.format(suf))
    for k, v in strategies.items():
        print('Node {}: util before is {} '.format(place_jp[k], utility_matrix[k]))
        # apply strategies (both back and forth)
        utility_matrix[k] += np.array(v)

        print('Node {}: util after is {} '.format(place_jp[k], utility_matrix[k]))

    # if run
    if run:
        eval_fun_trips(s_opt, _dir=dir, _suf=suf)

    predicted_trip_matrix = parse_trip_matrix(dir, suf)  # suffix is ''. no suffix
    predicted_trip_chains = parse_trip_pattern(dir, suf)

    return {'stg_no': suf, 'trip_matrix': predicted_trip_matrix, 'trip_chains': predicted_trip_chains}


def parse_tripchain_df(_predicted_df, *tdm_applied_dfs):
    res = _predicted_df.set_index('trip pattern')
    # change column names
    new_columns = res.columns.values
    new_columns[0], new_columns[1] = 'predicted pattern JP', 'predicted cnt'
    res.columns = new_columns

    for idx, df in enumerate(tdm_applied_dfs):
        df.set_index('trip pattern', inplace=True)
        # change column names
        new_columns = df.columns.values
        new_columns[0], new_columns[1] = 'JP of stg {}'.format(idx + 1), 'cnt of stg {}'.format(idx + 1)
        df.columns = new_columns
        # merge dfs
        res = pd.merge(res, df, left_index=True, right_index=True, how='outer')
        res = res.fillna(0)
        res['cnt_diff_{}'.format(idx + 1)] = res['cnt of stg {}'.format(idx + 1)] - res['predicted cnt']
        pass

    # replace nan values to zeros
    res = res.fillna(0)

    return res


if __name__ == '__main__':
    # Data preparation
    # %% read place code
    place_jp = pd.read_excel(os.path.join(os.path.dirname(__file__), 'slvr', 'Database', 'Place_code.xlsx')).name.values
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

    s = [0.032579037, 318.4565499, 0.1, 0.2]  # Feb. 2

    test_flag = input("1. Initiate penalty evaluation test? Any key to proceed 'Enter' to skip.")
    if test_flag:
        test_obj = eval_fun(s)
        print('Test penalty for beta* :', test_obj)

    # %% statistics of interest
    enumerated_usrs, obs_trip_matrix = parse_observed_trips(agent_database)
    obs_trip_chains = []
    for _ in agent_database:
        if _.preference is None or _.path_obs is None:
            continue
        # skip empty paths (no visited location)
        if len(_.path_obs) < 3:
            continue
        obs_trip_chains.append(list(np.array(_.path_obs) - 1))
    # observed trip tables 要不要scaled to the whole population's size?
    write_flag = 0
    if write_flag:
        pd.DataFrame(obs_trip_matrix).to_excel(
            'Project Database/Simulation Statistics/Observed trip frequency (PT only).xlsx')

    # %%  predicted trip tables
    s_opt = [0.006729682, 393.7222513, 0.859129711, 0.390907255]
    s_null = [0, s_opt[1], 0, 0]

    pdt_trip_dir = os.path.join(os.path.dirname(__file__), 'slvr', 'SimInfo', 'temp')
    if input('2. Evaluate predicted trip table given current optimal set of parameters? Enter to skip.'):
        # evaluate and dump
        eval_fun_trips(s_opt, _dir=pdt_trip_dir)

    predicted_trip_matrix = parse_trip_matrix(pdt_trip_dir, '')  # suffix is ''. no suffix
    predicted_trip_chains = parse_trip_pattern(pdt_trip_dir, '')

    write_flag = 0
    today = datetime.date.today().strftime('%m_%d')
    filename = 'Predicted trip table {}.xlsx'.format(today)
    if write_flag:
        pd.DataFrame(predicted_trip_matrix).to_excel(
            'Evaluation result/TDM simulation/{}'.format(filename))

    error_trip_table = predicted_trip_matrix - obs_trip_matrix
    # error_trip_percentage = (predicted_trip_tables - obs_trip_table) / obs_trip_table

    # draw comparison between observed and predicted node visit frequencies
    compare_plot = 1
    if compare_plot:
        compare_node_visit(obs_trip_chains, predicted_trip_chains['raw'])

    # %% simulation of TDM strategies
    """ 
    Scenario 1: travel impedance. 
    Strategies: a. reduce waiting time by increasing operation frequency (reduce cabin congestion as well); 
                b. introduce new or alternative lines
                c. reducing transit fare (same effect of travel time)
                
    Targets: congestion
             1. “Sanjusangen-do Temple” to “Gion”, 28-24
             2. “Ginkakuji area” to “Arashiyama region”,  15-23
             3. “Kyoto station area” to “Kiyomizu Temple area”,  29-27
             cabin
             -. “Arashiyama region” to “Gion”,  23-24
             frequency
             4. “Ginkakuji area” to “Gion” , 15-24  * all combinations among 15, 16, 17 24, that area 
             5. “Kyoto station area” to “Sanjusangen-do Temple”, 29-28
             fare:
             6. “Sagano Region” to “Kawaramachi area”,  14-25
    
    Empirical targets:
            1. Kyoto station to 高雄 , to 京北
            2. improve travel time of edges between 15, 16, 17, 24, 27 
    """

    trip_temp_dir = os.path.join(os.path.dirname(__file__), 'slvr', 'SimInfo', 'temp', 'time')  # directory

    effect = 0.2
    time_targets = [(27, 23), (14, 22), (28, 26), (14, 15, 16, 23), (28, 27), (13, 24)]

    # time_targets = [(27, 23), (14, 22), (28, 26)]
    # 可见一个strategy的list, 然后enumerate strategy instances

    strategy_set = [StrategyEdge('time', _, effect) for _ in time_targets]

    run_dummy = bool(input('3. Run tdm scenarios? Enter to skip.'))

    tdm_edge_res = []
    # evaluate (if run dummy is True) and parse
    for _ in strategy_set:
        tdm_edge_res.append(run_time_strategy(_.target, _.effect, trip_temp_dir, _.suffix, run=run_dummy))

    # plot tdm strategy effects
    _plot = 0
    if _plot:
        plot_visit_freq_effect(predicted_trip_chains['raw'], *tdm_edge_res[:3])
        plot_visit_freq_effect(predicted_trip_chains['raw'], *tdm_edge_res[3:])

    # create tdm effect trip chain df
    _create_df = 1
    predicted_df = predicted_trip_chains['no_OD']
    tdm_dfs = [_['trip_chains']['no_OD'] for _ in tdm_edge_res]
    if _create_df:
        res_df = parse_tripchain_df(predicted_df, *tdm_dfs)
        # res_df.to_excel('Comparison of tdm strategies edge (updated).xlsx')

    # %% predicted attraction visit frequency and tdm strategy effects plot
    new_edge_res = []

    """ instantiate strategies here if any new included. """

    # %%
    """ 
    Scenario 2: node attractiveness. 
    Strategies: increase attractiveness by introducing investment
    
    Targets: we also reviewed attraction areas that should have higher attractiveness 
      1) Empirical: 
              a. 1, 2, 35, 36, 37 (distant areas)
              b. 20, 21;  (Nijo area)
              c. 19, 22 (花園方面 (also increase 金阁寺 attradtivenss)
              d. 26, 30 ( Katsura area, potential package sites) (also increase 14和23 也要改） , 
              e. 31, 32, 33  (東福寺周辺, 東寺周辺, 伏見稲荷大社周辺 ) vs. [京都站和祇園
              f. 見直す review of the areas that should have higer value

    """

    trip_temp_dir = os.path.join(os.path.dirname(__file__), 'slvr', 'SimInfo', 'temp', 'util')  # directory

    type_1 = [0, 0.3, 0.3]  # improve in cultural and art + leisure activities (additive)
    type_2 = [0.3, 0.3, 0.3]
    type_3 = [0.3, 0, 0]

    # node indices start from 0
    node_strategies = {1: {0: type_1, 1: type_2, 34: type_2, 35: type_2, 36: type_2},
                       2: {19: type_1, 20: type_1},
                       3: {18: type_1, 21: type_1, 9: type_3},
                       4: {25: type_2, 29: type_2, 13: type_1, 22: type_3},
                       5: {30: type_2, 31: type_2, 32: type_2},
                       6: {14: type_3, 32: type_3, 9: type_3, 13: type_1, 22: type_3}
                       }

    strategy_set_node = [StrategyNode('util', v) for k, v in node_strategies.items()]

    run_dummy = bool(input('4. Run TDM node scenarios? Enter to skip.'))

    tdm_node_res = []
    # evaluate (if run dummy is True) and parse
    for _ in strategy_set_node:
        tdm_node_res.append(run_node_strategy(_.strategy, trip_temp_dir, _.suffix, run=run_dummy))

    # plot tdm node strategy effects
    _plot_node = 0
    if _plot_node:
        plot_visit_freq_effect(predicted_trip_chains['raw'], *tdm_node_res[:3])
        plot_visit_freq_effect(predicted_trip_chains['raw'], *tdm_node_res[3:])

    # create tdm effect trip chain df
    _create_df = 1
    tdm_node_dfs = [_['trip_chains']['no_OD'] for _ in tdm_node_res]
    if _create_df:
        res_df = parse_tripchain_df(predicted_df, *tdm_node_dfs)
        # res_df.to_excel('Comparison of tdm strategies node (updated).xlsx')
    # %% Sankey diagram for attraction area of interest

    # sankey(predicted_trip_chains['raw'], n - 1, place_jp)
    # compare between the predicted and observed
    try:
        while True:
            n = int(input('Please input the attraction index, starting from 1. E.g. node 1: 1:\n'))
            sk.sankey_echart_comparison(obs_trip_chains, predicted_trip_chains['raw'], n - 1, place_jp)

    except ValueError or IndexError:
        pass  # 有错误就直接pass了

    # %% compare between the base and improved predicted result
    stg_dfs = [tdm_edge_res[0]['trip_chains']['raw'],
               tdm_edge_res[3]['trip_chains']['raw'],
               tdm_node_res[1]['trip_chains']['raw'],
               tdm_node_res[3]['trip_chains']['raw']
               ]
    stg_names = ['Edge1', 'Edge 4', 'Node 2', 'Node 4']

    for _, new in enumerate(stg_dfs):
        print('\nCompare effects of strategy {}'.format(stg_names[_]))
        try:
            while True:
                n = int(input('  Please input the attraction index, starting from 1. E.g. node 1: 1. '
                              'Enter to skip.\n'))
                sk.sankey_echart_strategy(predicted_trip_chains['raw'], new, n - 1, place_jp, stg_names[_])
        except ValueError or IndexError:
            pass  # 有错误就直接pass了

    # %% Appendix and reference

    # %% Error between the utilities of observed trip and predicted trip, for each tourist using PT
    # if input('3. Evaluate utility tuples of observed and predicted trips? Enter to skip, any key to proceed.'):
    #     to_plot_tuples = eval_fun_util_tuples(s_opt)
    # else:
    #     tuple_temp_dir = os.path.join(os.path.dirname(__file__), 'slvr', 'SimInfo', 'temp', 'scatter plot')
    #     to_plot_tuples = parse_pdt_tuples(tuple_temp_dir)
    #
    # # scatter plot
    # scatter_plot(to_plot_tuples)
    #
    # #
    # if input('4. Evaluate utility tuples of observed and predicted for the null case?'):
    #     to_plot_tuples_null = eval_fun_util_tuples(s_null)
    #     # scatter plot
    #     scatter_plot(to_plot_tuples_null)
    # else:
    #     pass

    # maxfreq = n.max()
    # # 设置y轴的上限
    # plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

    # %%summarize areas into regions

    region = pd.read_excel(os.path.join(os.path.dirname(__file__), 'Project Database', 'Region partition.xlsx'))
    region_names = list(region['Region name'].values)
    region_areas = [str(_).split(',') for _ in region['Areas'].values]

    area_region_dict = {}
    for idx, _ in enumerate(region_names):
        for _node in region_areas[idx]:
            area_region_dict[int(_node) - 1] = idx

    region_compare_plot = 1
    if region_compare_plot:
        compare_region_visit(obs_trip_chains, predicted_trip_chains['raw'])
    # %% histogram of total visited attractions

    null_trip_dir = os.path.join(os.path.dirname(__file__), 'slvr', 'SimInfo', 'temp', 'null')
    if input('5. Evaluate predicted trip table at null case? Enter to skip.'):
        # evaluate and dump
        eval_fun_trips_null(s_null, _dir=null_trip_dir)

    null_trip_matrix = parse_trip_matrix(null_trip_dir, '')  # suffix is ''. no suffix
    null_trip_chains = parse_trip_pattern(null_trip_dir, '')

    if input('\nPlot histogram of total visited attractions?'):
        visited_cnt = []
        for _ in agent_database:
            visited_cnt.append(len(_.path_obs) - 2)

        #  matplotlib.axes.Axes.hist() 方法的接口
        # bins_range = np.array([_ for _ in range(max(visited_cnt) + 2)])

        bins_range = np.arange(0, max(visited_cnt) + 1.5) - 0.5

        fig, axes = plt.subplots(2, 1, dpi=200)
        n, bins, patches = axes[0].hist(x=visited_cnt, bins=bins_range,
                                        alpha=0.5, rwidth=0.85, label='observed')
        n, bins, patches = axes[1].hist(x=visited_cnt, bins=bins_range,
                                        alpha=0.5, rwidth=0.85, label='observed')

        visited_cnt_pdt = []
        for _ in predicted_trip_chains['raw']:
            visited_cnt_pdt.append(len(_) - 2)

        n, bins, patches = axes[0].hist(x=visited_cnt_pdt, bins=bins_range,
                                        alpha=0.5, rwidth=0.85, color='#0504aa', label='predicted')
        axes[0].set_title('Histogram of number of visits for the population')
        axes[0].set_ylabel('Frequency')
        axes[0].set_ylim(0, 500)
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.75)

        visited_cnt_null = []
        for _ in null_trip_chains['raw']:
            visited_cnt_null.append(len(_) - 2)

        n, bins, patches = axes[1].hist(x=visited_cnt_null, bins=bins_range,
                                        alpha=0.2, rwidth=0.85, color='orange', label='null')

        axes[1].set_ylabel('Frequency')
        axes[1].set_ylim(0, 500)
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.75)
        axes[1].set_xlabel('Places visited')

        plt.show()
