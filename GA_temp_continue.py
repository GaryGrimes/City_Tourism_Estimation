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
import csv

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

        # with gamma utility function
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

    PARAMETER[_itr] = _s  # save parameters of each iteration into the PARAMETER dict.

    scores = penalty2score(para_penalties)[0]  # functions returns ndarray

    # print evaluation scores
    print('Evaluation scores for iteration {}:'.format(_itr))
    for _i, _ in enumerate(scores):
        print('Parameter %d: a1: %.3f, b1: %.3f; k: %.3f, theta: %.3f, with score: %.3e'
              % (_i + 1, _s[_i][0], _s[_i][1], _s[_i][2], _s[_i][3], _))
    return scores


def selection(s_size, _scores):
    insertion_size = int(s_size / 5) + 1

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


def mutation_new(prob, best_score, population, population_scores, boundaries=(10, 1, 10, 3)):
    """ The mutation process after selection. An insersion of elite individuals will be performed as
    an elite preservation strategy.Last modified on Jan. 13, 2020."""

    if len(boundaries) != len(population[0]):
        raise ValueError('Bounds should have same number of entries as individuals.')
    # insert elite individuals
    insertion_size = int(len(population) / 3)

    # learning_rate = [0.01, 0.01, 0.01, 0.02]
    species = []

    # pick the bests
    population_sorted = [population[_] for _ in np.argsort(population_scores)]

    # pick the largest 5 individuals to perform
    for _index, _i in enumerate(population):
        do_mut_prob = np.random.rand()
        if do_mut_prob < prob:  # perform mutation, else pass
            _new_individual = []

            # compare the error between scores
            _score = population_scores[_index]
            _error = abs((_score - best_score) / best_score)

            for _idx, _value in enumerate(_i):
                bound = boundaries[_idx]
                _new_individual.append(value_mutation(_value, _error, bound))
            species.append(_new_individual)
        else:
            species.append(_i)

    # always preserve the individuals with high scores """
    species.extend(population_sorted[-insertion_size:])
    return species


def value_mutation(cur_value, error, bound):
    """Perform mutation to a current value with given mutation strength. Bound equals to learn rate."""
    value_mut_strength = 2 * (np.random.rand() - 0.5)  # a value between -1 and 1
    update_step = 0.01 * abs(bound)
    weight = 4 * error  # 0 <= weight < 5

    # purely random search outperforms converged search (referred to current optimal),
    # becasue current optimal means nothing...

    sway = value_mut_strength * ((1 + weight) * update_step)  # step (1, 5) of learn rate
    while cur_value * (cur_value + sway) < 0:
        value_mut_strength = 2 * (np.random.rand() - 0.5)  # a value between -1 and 1
        sway = value_mut_strength * ((1 + weight) * update_step)  # step (1, 5) of learn rate
    new_value = cur_value + sway
    return new_value


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
    # inn=16
    inn = 16  # species size (each individual in current generation is a vector of behavioral parameters

    # itr_max = 200
    itr_max = 100
    prob_mut = 1  # parameter mutation probability (always mutate to go random search)

    PARAMETER = {}  # save parameters in each iteration

    y_mean, y_max, x_max, gnr_max = [], [], [], []  # 记录平均score, 每一世代max score， 每世代最佳个体
    # generate first population

    initial_eval_filename = 'Initialization values final 01_29.xlsx'  # generated on Jan. 10

    initial_eval_res = pd.read_excel(
        os.path.join(os.path.dirname(__file__), 'Evaluation result', initial_eval_filename), index_col=0)

    # sort values by penalty
    temp_df = initial_eval_res.sort_values(by=['penalty'])
    s = temp_df.loc[:, 'a1':'scale'].values[:(inn)]


    filename = '{} iteration result.csv'.format(os.path.basename(__file__).split('.')[0])

    with open(os.path.join(os.path.dirname(__file__), 'Evaluation result', 'RandomGA', filename),
              'w', newline='') as csvFile:  # 去掉每行后面的空格
        fileHeader = ['itr', 'a1', 'intercept', 'shape', 'scale', 'penalty', 'score', 'record_penalty', 'record',
                      'gnr_mean']
        writer = csv.writer(csvFile)
        writer.writerow(fileHeader)

    # todo: read evaluated results into variables, and save into csv file
    temp_eval_filename = ' 填入文件名 '  # generated on Jan. 10

    temp_res = pd.read_excel(
        os.path.join(os.path.dirname(__file__), 'Evaluation result', 'RandomGA', temp_eval_filename), index_col=0)


    Res = pd.DataFrame(
        columns=['itr', 'a1', 'intercept', 'shape', 'scale', 'penalty', 'score', 'record_penalty', 'record',
                 'gnr_mean'])
    Res['itr'] = range(itr_max)
    Res.loc[:, 'a1':'scale'] = x_max
    Res['score'] = gnr_max
    Res['penalty'] = score2penalty(gnr_max)[0]
    Res['record_penalty'] = score2penalty(y_max)[0]
    Res['record'] = y_max
    Res['gnr_mean'] = y_mean


    # todo: write each iteration into csv file

    with open(os.path.join(os.path.dirname(__file__), 'Evaluation result', filename),
              'a', newline='') as csvFile:
        for _idx, row in enumerate(zip(para_penalties, scores)):
            add_info = [csv_index] + s[_idx] + list(row)
            writer = csv.writer(csvFile)
            writer.writerow(add_info)
            csv_index += 1

    # define mutation bounds for the parameters
    # bounds = np.array([max(abs(s[:, _])) for _ in range(s.shape[1])])
    bounds = np.array([0.1, 700, 7, 3])

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

        Best_score = max(SCORES)

        para_record = max(Best_score, para_record)  # duplicate 'record' use in the optimal solver module

        # write generation record and scores
        gnr_max.append(Best_score)
        y_mean.append(np.mean(SCORES))
        y_max.append(para_record)
        x_max.append(s[np.argsort(SCORES)[-1]])  # pick the last one with highest score. x_max

        # mutation to produce next generation
        s = mutation_new(prob_mut, para_record, s, SCORES,
                         boundaries=bounds)  # mutation generates (inn + insertion size) individuals

        print('\nMutated parameters(individuals) for iteration {}: '.format(iteration + 1))
        for i in range(len(s)):
            print('Parameter %d: a1: %.3f, b1: %.3f; k: %.3f, theta: %.3f\n' % (i + 1, s[i][0],
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

        # todo: save each iteration into csv file
    # %% save results into DF
    Res = pd.DataFrame(
        columns=['itr', 'a1', 'intercept', 'shape', 'scale', 'penalty', 'score', 'record_penalty', 'record',
                 'gnr_mean'])
    Res['itr'] = range(itr_max)
    Res.loc[:, 'a1':'scale'] = x_max
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
