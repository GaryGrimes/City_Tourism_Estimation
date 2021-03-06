"""This script is the main component in the research framework.
It is used to search for optimal behavioral parameter.
Dependence includes the 'slvr' package, data wrapping and so on.
Modified on Oct. 15. Last modified on Nov. 16"""

import numpy as np
import pickle
import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt
from slvr.SolverUtility_ILS import SolverUtility as ILS
import slvr.SimDataProcessing as sim_data
from slvr.SolverUtility_OP import SolverUtility as OP
import multiprocessing as mp

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


def selection(s_size, _scores):
    insertion_size = 3
    best_one_idx = np.argsort(_scores)[-1]
    f_sum = sum(_scores)
    prob = [_ / f_sum for _ in _scores]
    # calculate accumulated prob
    prob_acu = [sum(prob[:_]) + prob[_] for _ in range(len(prob))]
    prob_acu[-1] = 1

    # return selected idx
    indices = []
    for _ in range(s_size - insertion_size):
        random_num = np.random.rand()
        indices.append(next(_x[0] for _x in enumerate(prob_acu) if _x[1] > random_num))  # x is a tuple
    # insert best results from history
    indices.extend(insertion_size * [best_one_idx])
    return indices


# todo: mutation process is to be modified.
def mutation(prob, best_score, population, population_scores):
    insertion_size = round(len(population) / 3)
    learn_rate = [0.01, 0.01, 0.01, 0.02]
    species = []
    best = list(population[np.argsort(population_scores)[-1]])

    # pick the largest 5 individuals to perform
    for _index, _i in enumerate(population):
        mut_temp = np.random.rand()
        if mut_temp < prob:  # perform mutation, else pass
            _score = population_scores[_index]
            weight = 4 * (np.abs(_score - best_score) / best_score)  # 0 <= weight < 5
            _new_individual = []
            # alphas should < 0
            for _j, _par_a in enumerate(_i[:2]):
                _gain = 2 * (np.random.rand() - 0.5) * ((1 + weight) * learn_rate[_j])  # step (1, 5) of learn rate
                # proportional to the parameter size
                while _par_a + _gain > 0:
                    _gain = 2 * (np.random.rand() - 0.5) * ((1 + weight) * learn_rate[_j])
                _par_a += _gain  # update parameter
                _new_individual.append(_par_a)
            # betas should >= 0
            for _k, _par_b in enumerate(_i[2:]):
                _gain = 2 * (np.random.rand() - 0.5) * ((1 + weight) * learn_rate[2 + _k])  # step (1, 5) of learn rate
                # proportional to the parameter size
                while _par_b + _gain < 0:
                    _gain = 2 * (np.random.rand() - 0.5) * ((1 + weight) * learn_rate[2 + _k])
                _par_b += _gain  # update parameter
                _new_individual.append(_par_b)
            species.append(_new_individual)
        else:
            species.append(_i)
    # insert the best solution so far
    """ always preserve the best solution """
    species.extend(insertion_size * [best])
    return species


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

    # parameter setup
    inn = 12  # species size (each individual in current generation is a vector of behavioral parameters)
    itr_max = 200
    prob_mut = 1  # parameter mutation probability
    """ memo is to store parameter --> Path similarity objective function"""

    memo_parameter, memo_penalty = [], []  # memo stores parameters from last 2 iterations
    PARAMETER = {}

    # s = []  # species set of parameters

    # test s with fixed input
    s = [
        [-0.37974139765941783,
         -0.2558939179688775,
         0.04893578316589828,
         0.05210324618171788],
        [-0.08918215758059533,
         -0.24405051342668937,
         0.027020415460782002,
         0.2056606245278888],
        [-0.16167280374767834,
         -0.2411843920976503,
         0.03404410015008346,
         0.3076044553748146],
        [-0.46797554664851887,
         -0.08691471688216373,
         0.27465618122012814,
         0.8535210297561443],
        [-0.16654700848822268,
         -0.0887516253882134,
         0.14708878950043483,
         0.3303207960587167],
        [-0.3236607278310718,
         -0.0668914251165349,
         0.19367692132502703,
         0.4580954274520535],
        [-0.3232875461086227,
         -0.028247387947372693,
         0.24030778479735282,
         0.3322569213448343],
        [-0.059277941835195136,
         -0.4592104661803278,
         0.241806890829197,
         0.43319110214340956]
    ]

    # generate first population
    """ beta 1 is fixed here"""

    for i in range(inn - len(s)):  # to fill in s
        # random alphas
        a1, a2 = np.random.uniform(-0.5, -0.01), np.random.uniform(-0.5, -0.01)
        # random betas
        b2, b3 = np.random.uniform(0.01, 0.3), np.random.uniform(0.02, 1)
        s.append([a1, a2, b2, b3])
    # manually insert reasonable parameters
    s[-2], s[-1] = [-0.05, -0.05, 0.03, 0.1], [-0.03, -0.01, 0.02, 0.1]

    y_mean, y_max, x_max, gnr_max = [], [], [], []  # 记录平均score, 每一世代max score， 每世代最佳个体

    # calculate score and record of the 1st generation
    print('------ Solver utility debug: enumeration size and execution time ------')

    # print('------ Initialization: first generation ------')

    # evaluate scores for the first generation
    jobs = []
    PenaltyQueue = mp.Queue()  # queue, to save results for multi_processing
    # PENALTIES = {}  # queue, to save results for multi_processing

    # calculate evaluation time
    start_time = datetime.datetime.now()

    # execution time test for ILS
    # test_parameter = [-0.05, -0.05, 0.03, 0.1]

    # test_parameter = [-1.286284872, -0.286449175, 0.691566901, 0.353739632]

    test_parameter = s[0]

    ALPHA = test_parameter[:2]
    BETA = [5] + test_parameter[2:]
    data_input = {'alpha': ALPHA, 'beta': BETA,
                  'phi': phi,
                  'util_matrix': utility_matrix,
                  'time_matrix': edge_time_matrix,
                  'cost_matrix': edge_cost_matrix,
                  'dwell_matrix': dwell_vector,
                  'dist_matrix': edge_distance_matrix}

    # start process
    idx = 1
    if input("Run solver debug? 'Enter' to skip."):
        res_ILS = ILS.solver_debug(idx, node_num, agent_database, **data_input)

        # res_OP = OP.solver_debug(idx, node_num, agent_database, **data_input)

        end_time = datetime.datetime.now()
        print('------ Evaluation time: {}s ------\n'.format((end_time - start_time).seconds))

        # %% error types debug
        agents_empty_info = [agent_database[_] for _ in res_ILS['error_emty_info']]
        agents_no_init = [agent_database[_] for _ in res_ILS['error_init']]

        # 10个tourist没有observed path、只有o和d.默认跳过
        agents_no_path = [agent_database[_] for _ in res_ILS['error_no_path']]

    # todo debug procedures
    # %% test for the # of successful enumerated tourists by mp
    if input("Run debug mp? 'Enter' to skip."):
        s[-1] = test_parameter
        for idx, parameter in enumerate(s):
            print('Starting process {} in {}'.format(idx + 1, inn))

            # check existence of parameter in memory
            if parameter in memo_parameter:
                PenaltyQueue.put((idx, memo_penalty[memo_parameter.index(parameter)]))
                print('The {}th parameter is sent from history (with index {}), with score: {}'.format(idx,
                                                                                                       memo_parameter.index(
                                                                                                           parameter),
                                                                                                       memo_penalty[
                                                                                                           memo_parameter.index(
                                                                                                               parameter)]))
            else:
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
                process = mp.Process(target=ILS.solver_debug_mp, args=(PenaltyQueue, idx, node_num, agent_database),
                                     kwargs=data_input, name='P{}'.format(idx + 1))
                jobs.append(process)
                process.start()

        for j in jobs:
            # join process
            j.join()

        # 从 queue里提取parameter penalties
        Para_penalties_tuples = []
        while True:
            if PenaltyQueue.empty():  # 如果队列空了，就退出循环
                break
            else:
                Para_penalties_tuples.append(PenaltyQueue.get())

        Para_penalties = []
        for _i in range(len(s)):
            for _tuple in Para_penalties_tuples:
                if _i == _tuple[0]:
                    Para_penalties.append(_tuple[1])
                    break

        SCORES = list(penalty2score(Para_penalties)[0])  # functions returns ndarray

        # print evaluation scores
        # print('Evaluation scores:')
        # for i, _ in enumerate(SCORES):
        #     print(
        #         'Parameter %d: a1: %.3f, a2: %.3f; b2: %.3f, b3: %.3f, with score: %.3e' % (i + 1, s[i][0],
        #                                                                                     s[i][1],
        #                                                                                     s[i][2],
        #                                                                                     s[i][3],
        #                                                                                     _))
