import numpy as np
import pickle
import datetime
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
# from slvr.SolverUtility_OP import SolverUtility
from slvr.SolverUtility_ILS import SolverUtility
import multiprocessing as mp
import progressbar as pb

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class Agent(object):
    agent_count = 0

    def __init__(self, index, uid):
        self.index, self.uid = index, uid
        self.trip_df, self.time_budget = None, 0
        self.path_pdt, self.path_obs = None, None
        self.preference = None
        Agent.agent_count += 1


class Tourist(object):
    tourist_count = 0

    def __init__(self, index, uid):
        self.index, self.uid = index, uid
        self.trip_df, self.time_budget = None, 0
        self.path_pdt, self.path_obs = None, None
        self.preference = None
        Tourist.tourist_count += 1


def print_path(path_to_print):
    print(list(np.array(path_to_print) + 1))


def penalty(particle):
    _answer = [-0.02, -0.01, 0.3, 0.1]
    diff = np.array(_answer) - np.array(particle)
    _penalty = np.exp(np.linalg.norm(diff))
    return _penalty


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


if __name__ == '__main__':
    # %% Solver Setup
    # load agents
    with open(os.path.join(os.path.dirname(__file__), 'slvr',
                           'Database', 'transit_user_database.pickle'), 'rb') as file:
        agent_database = pickle.load(file)

    print('Setting up agents...')

    # %% setting up nodes
    node_num = 37  # Number of attractions. Origin and destination are excluded.

    Intrinsic_utilities = pd.read_excel(
        os.path.join(os.path.dirname(__file__), 'slvr', 'Database', 'Intrinsic Utility.xlsx'),
        sheet_name='data')
    utility_matrix = []
    for _idx in range(Intrinsic_utilities.shape[0]):
        temp = np.around(list(Intrinsic_utilities.iloc[_idx, 1:4]), decimals=3)
        utility_matrix.append(temp)
    utility_matrix = np.array(utility_matrix)

    Dwell_time = pd.read_excel(os.path.join(os.path.dirname(__file__), 'slvr', 'Database', 'Dwell time array.xlsx'),
                               index_col=0)
    # replace missing values by average of all samples
    Dwell_time.loc[35, 'mean'] = Dwell_time['mean'][Dwell_time['mean'] != 5].mean()  # Attraction 35
    dwell_vector = np.array(Dwell_time['mean'])

    # %% edge property
    Edge_time_matrix = pd.read_excel(
        os.path.join(os.path.dirname(__file__), 'slvr', 'Database', 'Trips', 'Final', 'transit_time_update2.xlsx'),
        index_col=0)

    edge_time_matrix = np.array(Edge_time_matrix)

    # Edge travel time
    # need several iterations to make sure direct travel is shorter than any detour

    no_update, itr = 0, 0
    print('Starting travel_time_check...')
    for _ in range(3):
        while not no_update:
            print('Current iteration: {}'.format(itr + 1))
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
                print('Travel time update complete.\n')

    # Edge travel cost (fare)
    Edge_cost_matrix = pd.read_csv(
        os.path.join(os.path.dirname(__file__), 'slvr', 'Database', 'Trips', 'Final', 'wide_transit_fare_matrix.csv'),
        index_col=0)
    # Edge travel distance
    Edge_distance_matrix = pd.read_excel(
        os.path.join(os.path.dirname(__file__), 'slvr', 'Database', 'Trips', 'Final',
                     'driving_wide_distance_matrix.xlsx'),
        index_col=0)

    edge_cost_matrix = np.array(Edge_cost_matrix)
    # distance matrix for path penalty evaluation
    edge_distance_matrix = np.array(Edge_distance_matrix)  # distance between attraction areas

    #  check UtilMatrix等各个matrix的shape是否正确。与NodeNum相符合
    if len(utility_matrix) != node_num:
        raise ValueError('Utility matrix error.')
    if edge_time_matrix.shape[0] != edge_time_matrix.shape[1]:
        raise ValueError('Time matrix error.')
    if edge_cost_matrix.shape[0] != edge_cost_matrix.shape[1]:
        raise ValueError('Cost matrix error.')
    if len(dwell_vector) != node_num:
        raise ValueError('Dwell time array error.')

    print('Setting up solver.')
    # parameter setup
    phi = 0.1

    core_process = mp.cpu_count()  # species size (each individual is our parameters here)
    itv = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    size_itv = len(itv)
    # create parameter columns

    # create initial values of parameters
    indices = [[(i // size_itv ** 3) % 7, (i // size_itv ** 2) % 7, (i // size_itv) % 7, i % 7] for i in
               range(size_itv ** 4)]
    s = []

    Population = [[itv[indices[j][i]] for i in range(4)] for j in range(len(indices))]

    # alpha should have negative values
    for _ in Population:
        for j in range(len(_)):
            if j < 2 and _[j] > 0:
                _[j] = -_[j]

    # calculate score and record of the 1st generation
    time, itr = 0, 0
    Population_penalties, Population_scores = [], []

    # set up progress bar
    progress = pb.ProgressBar(widgets=[
        ' [', pb.Timer(), '] ',
        pb.Percentage(),
        ' (', pb.ETA(), ') ',
    ])

    progress.start()
    pg_total = len(indices)  # for progressbar need to pre-calculate the total and current index

    while Population:
        itr += 1
        print('------ Evaluation start for iteration {} ------\n'.format(itr))

        try:
            pg_curr = pg_total - len(Population)
            progress.update(int(pg_curr / (pg_total - 1)) * 100)
        except:
            pass

        s = Population[:core_process]

        jobs = []
        penalty_queue = mp.Queue()  # queue, to save results for multi_processing

        # calculate evaluation time
        start_time = datetime.datetime.now()

        for idx, parameter in enumerate(s):
            print('Starting process {} in {}'.format(idx + 1, len(s)))

            ALPHA = parameter[:2]
            BETA = [5] + parameter[2:]
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

        for j in jobs:
            # join process
            j.join()

        end_time = datetime.datetime.now()
        print('------ Evaluation time for iteration {} : {}s ------\n'.format(itr, (end_time - start_time).seconds))

        time += (end_time - start_time).seconds
        print('------ Total time passed: {} hh {} mm {} ss ------\n'.format(time // 3600,
                                                                            time % 3600 // 60,
                                                                            time % 60))
        # 从 queue里取值
        Para_penalties_tuples = []
        while True:
            if penalty_queue.empty():  # 如果队列空了，就退出循环
                break
            else:
                Para_penalties_tuples.append(penalty_queue.get())

        para_penalties = []
        for _i in range(len(s)):
            for _tuple in Para_penalties_tuples:
                if _i == _tuple[0]:
                    para_penalties.append(_tuple[1])
                    break

        scores = list(penalty2score(para_penalties)[0])  # functions returns ndarray

        # write generation record and scores
        Population_penalties.extend(para_penalties)
        Population_scores.extend(scores)

        # print evaluation scores
        print('Evaluation scores:')
        for i, _ in enumerate(scores):
            print(
                'Parameter %d: a1: %.3f, a2: %.3f; b2: %.3f, b3: %.3f, with score: %.3e' % (i + 1, s[i][0],
                                                                                            s[i][1],
                                                                                            s[i][2],
                                                                                            s[i][3],
                                                                                            _))

        del Population[:core_process]

    # %% save results into DF
    Population = [[itv[indices[j][i]] for i in range(4)] for j in range(len(indices))]
    Res = pd.DataFrame(columns=['index', 'a1', 'a2', 'b2', 'b3', 'penalty', 'score'])
    Res['index'] = range(len(Population))
    Res.loc[:, 'a1':'b3'] = Population
    Res['score'] = Population_scores
    Res['penalty'] = Population_penalties

    file_name = 'ILS_PF_LD'  # ILS, with path threshold (filtering), with levenshtein distance
    Res.to_excel('Initialization objective values {}.xlsx'.format(file_name))

# %%  todo build the queue check process, to read from queue as soon as it fills, so it never gets very large.
#
#         Time_threshold = 480  # 一般每个process至少大于八分钟吧？
#         start = time.time()  # in seconds
#         while time.time() - start >= Time_threshold:
#             if not any(p.is_alive() for p in jobs):  # jobs is the list of current processes
#                 # All the processes are done, break now.
#                 break
#             # 任意一个process在跑的话
#         # ------------ 工事中 ------------ #
#             '''工事中。'''
#             PENALTIES.get()
#
#             '''工事中'''
#
#         # ------------ 工事中 ------------ #
#
#             time.sleep(1)  # 停一秒
