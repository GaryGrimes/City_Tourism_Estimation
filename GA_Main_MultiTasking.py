"""This script is the main functionality in the research framework, used to search for optimal behavioral parameters.
Compared with Evolutionary Strategy, this script utilizes random search GA for optimal search.
Dependence includes the 'slvr' package, data wrapping and so on. Modified on Oct. 15. Last modified on Nov. 16
Last modified on Jan. 11. Current code duplicates with ES_Main. Will be merged with MultiTasking.py.
"""

import numpy as np
import pickle
import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt
from slvr.SolverUtility_ILS import SolverUtility
import multiprocessing as mp
import slvr.SimDataProcessing as sim_data

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
    ''' just for test '''
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


def evaluation(_s, _itr):
    """Evaluation of each population using MultiProcessing. Results are returned to the mp.queue in tuples."""
    global PARAMETER
    global memo_parameter, memo_penalty
    global phi, utility_matrix, dwell_vector, edge_time_matrix, edge_cost_matrix, edge_distance_matrix

    print('------ Iteration {} ------\n'.format(_itr + 1))

    jobs = []
    penalty_queue = mp.Queue()  # queue, to save results for multi_processing

    # calculate evaluation time
    start_time = datetime.datetime.now()

    # evaluation with MultiProcessing for each parameter in current generation
    for idx, parameter in enumerate(_s):
        print('\nStarting process {} in {}'.format(idx + 1, len(_s)))
        # convert ndarray to list
        try:
            parameter = parameter.tolist()  # convert to list
        except AttributeError:
            pass

        # check existence of parameter in memory
        if parameter in memo_parameter:
            # sent back penalty tuple if exists in history
            penalty_queue.put((idx, memo_penalty[memo_parameter.index(parameter)]))
            print('\nThe {}th parameter is sent from history (with index {}), with score: {}'.format(
                idx, memo_parameter.index(parameter), memo_penalty[memo_parameter.index(parameter)]))
        else:
            ALPHA = list(parameter[:2])
            BETA = [100] + list(parameter[2:])
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


def selection(s_size, _scores):
    insertion_size = np.floor(s_size / 5) + 1

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


# def selection(s_size, _scores):
#     insertion_size = 3
#     best_one_idx = np.argsort(_scores)[-1]
#     f_sum = sum(_scores)
#     prob = [_ / f_sum for _ in _scores]
#     # calculate accumulated prob
#     prob_acu = [sum(prob[:_]) + prob[_] for _ in range(len(prob))]
#     prob_acu[-1] = 1
#
#     # return selected idx
#     indices = []
#     for _ in range(s_size - insertion_size):
#         random_num = np.random.rand()
#         indices.append(next(_x[0] for _x in enumerate(prob_acu) if _x[1] > random_num))  # x is a tuple
#     # insert best results from history
#     indices.extend(insertion_size * [best_one_idx])
#     return indices

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
    # %% Solver Setup
    # read tourist agents
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

    # parameter setup
    inn = 16  # species size (each individual in current generation is a vector of behavioral parameters
    itr_max = 200
    prob_mut = 1  # parameter mutation probability (always mutate to go random search)

    memo_parameter, memo_penalty = [], []  # memo stores parameters from last 2 iterations
    PARAMETER = {}  # save parameters in each iteration

    y_mean, y_max, x_max, gnr_max = [], [], [], []  # 记录平均score, 每一世代max score， 每世代最佳个体
    # generate first population

    # generate first population. s = []  # species set of parameters
    # parameter = [-0.05, -0.05, 0.03, 0.1]
    # s = [[]]  # todo initialize s with good results in initialization evaluation

    initial_eval_filename = 'Initialization objective values ILS_PF_NewLD.xlsx'  # generated on Jan. 10
    initial_eval_res = pd.read_excel(
        os.path.join(os.path.dirname(__file__), 'Evaluation result', initial_eval_filename), index_col=0)

    # sort values by penalty
    temp_df = initial_eval_res.sort_values(by=['penalty'])
    s = temp_df.loc[:, 'a1':'b3'].values[:(inn)]

    # alpha1 and alpha2 should have negative values!
    s[:, :2] = -s[:, :2]
    s = s.tolist()

    # start iterations
    para_record = float('-inf')
    for iteration in range(itr_max):
        # evaluation
        SCORES = evaluation(s, iteration)  # iteration 0 = initialization evaluation

        # selection
        Indices = selection(inn, SCORES)

        s = list(s[_] for _ in Indices)
        # Scores of selected individuals
        SCORES = list(SCORES[_] for _ in Indices)  # s and SCORES should have same dimension

        print('Iteration {}: SCORES:\n'.format(iteration + 1))
        for i, _ in enumerate(Indices):
            print(
                'Parameter %d: a1: %.3f, a2: %.3f; b2: %.3f, b3: %.3f, with score: %.3e' % (i + 1, s[i][0],
                                                                                            s[i][1],
                                                                                            s[i][2],
                                                                                            s[i][3],
                                                                                            SCORES[i]))

        Best_score = max(SCORES)

        para_record = max(Best_score, para_record)  # duplicate 'record' use in the optimal solver module

        # write generation record and scores
        gnr_max.append(Best_score)
        y_mean.append(np.mean(SCORES))
        y_max.append(para_record)
        x_max.append(s[np.argsort(SCORES)[-1]])  # pick the last one with highest score. x_max

        # mutation to produce next generation
        s = mutation(prob_mut, para_record, s, SCORES)  # mutation generates (inn + insertion size) individuals

        print('\nMutated parameters(individuals) for iteration {}: '.format(iteration + 1))
        for i in range(len(s)):
            print(
                'Parameter %d: a1: %.3f, a2: %.3f; b2: %.3f, b3: %.3f\n' % (i + 1, s[i][0],
                                                                            s[i][1],
                                                                            s[i][2],
                                                                            s[i][3],
                                                                            ))
        # %% plot
        if iteration > 20:
            try:
                x = range(iteration)
                fig = plt.figure(dpi=150)
                ax = fig.add_subplot(111)
                ax.plot(y_mean, color='lightblue')
                ax.plot(y_max, color='xkcd:orange')
                ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
                plt.ylabel("score")
                plt.xlabel('iteration')
                plt.xlim([0, iteration + 1])
                plt.title("Parameter update process")
                plt.legend(['average', 'best'], loc='lower right', shadow=True, ncol=2)
                plt.show()
            except:
                pass

    # %% save results into DF
    Res = pd.DataFrame(
        columns=['itr', 'a1', 'a2', 'b2', 'b3', 'penalty', 'score', 'record_penalty', 'record', 'gnr_mean'])
    Res['itr'] = range(itr_max)
    Res.loc[:, 'a1':'b3'] = x_max
    Res['score'] = gnr_max
    Res['penalty'] = score2penalty(gnr_max)[0]
    Res['record_penalty'] = score2penalty(y_max)[0]
    Res['record'] = y_max
    Res['gnr_mean'] = y_mean
    Res.to_excel('{} iteration result.xlsx'.format(os.path.basename(__file__).split('.')[0]))

    # save parameters into DF
    for k, v in PARAMETER.items():
        PARAMETER[k] = v.tolist()
    df_parameter = pd.DataFrame.from_dict(PARAMETER, orient='index')
    for i in range(df_parameter.shape[0]):
        for j in range(df_parameter.shape[1]):
            if df_parameter.loc[i, j]:
                df_parameter.loc[i, j] = [round(_, 3) for _ in np.array(df_parameter.loc[i, j])]

    df_parameter.to_excel('Parameter in each iteration.xlsx')
