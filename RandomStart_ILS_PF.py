import numpy as np
import pickle
import datetime
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
# from slvr.SolverUtility_OP import SolverUtility
from slvr.SolverUtility_ILS import SolverUtility
import multiprocessing as mp
import progressbar as pb
import slvr.SimDataProcessing as sim_data

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
    #  read tourist agents
    with open(os.path.join(os.path.dirname(__file__), 'slvr', 'Database', 'transit_user_database.pickle'),
              'rb') as file:
        agent_database = pickle.load(file)  # note: agent = tourists here

    print('Setting up agents...')

    """Test geo distance penalty function """
    today = datetime.date.today().strftime('%m_%d')
    filename = 'Initialization values GeoDist {}.csv'.format(today)

    with open(os.path.join(os.path.dirname(__file__), 'Evaluation result', filename),
              'w', newline='') as csvFile:  # 去掉每行后面的空格
        fileHeader = ['index', 'a1', 'intercept', 'shape', 'scale', 'penalty', 'score']
        writer = csv.writer(csvFile)
        writer.writerow(fileHeader)
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

    core_process = mp.cpu_count()  # species size (each individual is our parameters here)
    '''First random search'''
    # ranges for modified utility (gamma)
    # range_alpha = [-0.1, -0.3, -1, -3, -10, -30, -100]
    # range_intercept = [10, 30, 100, 300, 1000, 3000, 10000]
    # range_scale = [0.2, 0.3, 0.5, 0.8, 1.0]
    # exp_x = [3.5]
    '''Second random search'''
    # range_alpha = [-3, -10, -30]
    # range_intercept = [10, 30, 100]
    # range_shape = [3.5, 7, 35]
    #
    # range_exp_x = [1, 2, 2.5, 3, 4, 6, 8]
    """Jan. 26"""
    # range_alpha = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]
    # # intercept should be more than 50 times of alpha
    # range_intercept = [100, 300, 1000]
    # range_shape = [0.1, 0.5, 1, 2, 5, 7]
    # range_scale = [0.2, 0.4, 0.6, 0.8, 1, 2, 5]

    range_alpha = [0.03]
    # intercept should be more than 50 times of alpha
    range_intercept = [300]
    range_shape = [0.1, 0.5, 1, 2, 5, 7]
    range_scale = [0.2, 0.4, 0.6, 0.8, 1, 2, 5]

    # %% generate population

    Population = [[np.random.normal(i, 0.2 * i), np.random.normal(j, 0.1 * j), p, q]
                  for i in range_alpha for j in range_intercept for p in range_shape for q in range_scale]

    s = []

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
    pg_total = len(Population)  # for progressbar need to pre-calculate the total and current index

    csv_index = 0
    while Population:
        itr += 1
        print('------ Evaluation start for iteration {} ------\n'.format(itr))

        try:
            pg_curr = pg_total - len(Population) + core_process
            cur_progress = int(pg_curr / (pg_total - 1)) * 100
            progress.update(cur_progress)
        except:
            pass

        s = Population[:core_process]  # 因为无法pop掉多个

        jobs = []
        penalty_queue = mp.Queue()  # queue, to save results for multi_processing

        # calculate evaluation time
        start_time = datetime.datetime.now()

        for idx, parameter in enumerate(s):
            print('Starting process {} in {}'.format(idx + 1, len(s)))

            ALPHA = parameter[0]

            BETA = {'intercept': parameter[1], 'shape': parameter[2], 'scale': parameter[3]}
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

        # write into csv file

        with open(os.path.join(os.path.dirname(__file__), 'Evaluation result', filename),
                  'a', newline='') as csvFile:
            for _idx, row in enumerate(zip(para_penalties, scores)):
                add_info = [csv_index] + s[_idx] + list(row)
                writer = csv.writer(csvFile)
                writer.writerow(add_info)
                csv_index += 1

        del Population[:core_process]

    # # %% save results into DF
    # Population = [[i, j, p, q]
    #               for i in range_alpha for j in range_intercept for p in range_shape for q in range_scale]
    # # Res = pd.DataFrame(columns=['index', 'a1', 'a2', 'b2', 'b3', 'penalty', 'score'])
    # Res = pd.DataFrame(columns=['index', 'a1', 'intercept', 'shape', 'scale', 'penalty', 'score'])
    # Res['index'] = range(len(Population))
    # Res.loc[:, 'a1':'scale'] = Population
    # Res['score'] = Population_scores
    # Res['penalty'] = Population_penalties
    #
    # file_name = 'final formulation'  # ILS, with path threshold (filtering), with levenshtein distance
    # Res.to_excel('Initialization objective values {}.xlsx'.format(file_name))

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
