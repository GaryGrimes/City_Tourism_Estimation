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
from matplotlib import rcParams
import datetime

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


def eval_fun_trips(para, _dir_name=''):
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

        process = mp.Process(target=SolverUtility.solver_trip_stat,
                             args=(penalty_queue, idx, node_num, chunk, _dir_name),
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
            # Aggregate predicted trips from files in designated directory
            name = '{}predicted_trip_table_{}.pickle'.format(_dir_name, cur_idx)
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


# simulation strategies
def tdm_simulation(_taget_property, _strategy, _target_ODs):
    stg_type = ['time', 'cost', 'util', 'combined']
    # example
    directory_name = _strategy + '/'

    pass


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


def plot_node_visit(_original_trips, _after_trips):
    # grouped bar plot see:
    # https://matplotlib.org/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py

    node_visits_before = sum(_original_trips)

    visits_sorted_before, idx_sorted = np.sort(node_visits_before)[::-1], np.argsort(node_visits_before)[::-1]
    place_labels_jp = [place_jp[_] for _ in idx_sorted]

    # todo: when get the node visits after strategies, need to retrieve values according to the sorted indices: idx_sorted
    node_visits_after = sum(_after_trips)
    visits_sorted_after = [node_visits_after[_] for _ in idx_sorted]

    x = np.arange(len(place_labels_jp))  # the label locations
    width = 0.35  # the width of the bars

    fig_wide = 12
    fig, ax = plt.subplots(figsize=(fig_wide, fig_wide / 1.5), dpi=400)

    rects1 = ax.bar(x - width / 2, visits_sorted_before, width, label='Before')
    rects2 = ax.bar(x + width / 2, visits_sorted_after, width, label='After')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Visit frequency')
    ax.set_title('Visit frequency of places (before and after)')
    ax.set_xticks(x)
    ax.set_xticklabels(place_labels_jp)
    ax.legend()

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
    plt.xticks(rotation=75)
    fig.tight_layout()

    plt.grid(True, which='both', axis='y', linestyle="-.")
    plt.show()

    pass


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
    enumerated_usrs, obs_trip_table = parse_observed_trips(agent_database)

    # observed trip tables 要不要scaled to the whole population's size?
    write_flag = 0
    if write_flag:
        pd.DataFrame(obs_trip_table).to_excel(
            'Project Database/Simulation Statistics/Observed trip frequency (PT only).xlsx')

    # predicted trip tables
    s_opt = [0.032579037, 318.4565499, 0.1, 0.2]
    s_null = [0, s_opt[1], 0, 0]

    if input('2. Evaluate predicted trip table given current optimal set of parameters? Enter to skip.'):
        predicted_trip_tables = eval_fun_trips(s_opt)
    else:
        trip_temp_dir = os.path.join(os.path.dirname(__file__), 'slvr', 'SimInfo', 'temp')
        predicted_trip_tables = parse_pdt_trip(trip_temp_dir)

    write_flag = 0
    today = datetime.date.today().strftime('%m_%d')
    filename = 'Predicted trip table {}.xlsx'.format(today)
    if write_flag:
        pd.DataFrame(predicted_trip_tables).to_excel(
            'Evaluation result/TDM simulation/{}'.format(filename))

    error_trip_table = predicted_trip_tables - obs_trip_table
    error_trip_percentage = (predicted_trip_tables - obs_trip_table) / obs_trip_table

    original_trip_tables = predicted_trip_tables.copy()
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
             4. “Arashiyama region” to “Gion”,  23-24
             frequency
             5. “Ginkakuji area” to “Gion” , 6-24
             6. “Kyoto station area” to “Sanjusangen-do Temple”, 29-28
             fare:
             7. “Sagano Region” to “Kawaramachi area”,  14-25
    
    Empirical targets:
            1. Kyoto station to 高雄 , to 京北
            2. improve travel time of edges between 15, 16, 17, 24, 27 
    """
    # travel time
    strategy_type = 'time'
    targets = [(27, 23), (14, 22), (28, 26)]  # [(29, 24), (31,33)]
    effects = [0.2, 0.2, 0.2]  # reduce 20% of the travel time  range(0.2, 0.3, 0.5)

    for idx, _ in enumerate(targets):
        print('Target {} for travel time scenarios: {} to {}'.format(idx + 1, place_jp[_[0]], place_jp[_[1]]))
        print('travel_time_before is {:.3f} min(s)'.format(edge_time_matrix[_]))

        # todo: modify travel time, for both the back and forth tuples, e.g. (27, 23) and (23, 27)
        # todo: tdm_trips = eval_fun_trips(s_opt, _dir_name='time/')

    # %%
    """ 
    Scenario 2: node attractiveness. 
    Strategies: increase attractiveness by introducing investment
    
    Targets: 1. “Sanjusangen-do Temple” to “Gion”, 28-24
             2. “Ginkakuji area” to “Arashiyama region”,  15-23
             3. “Kyoto station area” to “Kiyomizu Temple area”.  29-27
    Empirical: 
              a. 1, 2, 35, 36, 37 (distant areas)
              b. 20, 21; 18, 17  (near Shijo Kawaramachi)
              c. 19, 22, 26, 30 ( near Arashiyama, package sites)
              d. 31, 32  (

            
    """


    # %% result and indicators
    """1. trip tables (after strategy) 2. effects in ratio (referring to the original trip statistics 
    3. attraction visit distribution"""

    # methods for plot attraction visits finished

    # todo: trip statistics before and after applying strategies
    # todo: compare predicted_trip_tables (after applying strategy) and original_trip_tables
    per_trip_change = predicted_trip_tables / original_trip_tables

    # %% histogram of total visited attractions
    if input('\nPlot histogram of total visited attractions?'):
        visited_cnt = []
        for _ in agent_database:
            visited_cnt.append(len(_.path_obs) - 2)

        #  matplotlib.axes.Axes.hist() 方法的接口
        # bins_range = np.array([_ for _ in range(max(visited_cnt) + 2)])

        bins_range = np.arange(0, max(visited_cnt) + 1.5) - 0.5

        fig, ax = plt.subplots(dpi=200)
        n, bins, patches = ax.hist(x=visited_cnt, bins=bins_range, color='#0504aa',
                                   alpha=0.6, rwidth=0.85)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Places visited')
        plt.ylabel('Frequency')
        plt.title('An histogram of total places visited')

        plt.show()
