import numpy as np
import numdifftools as nd
import os, pickle
import multiprocessing as mp
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import slvr.SimDataProcessing as sim_data
from slvr.SolverUtility_ILS import SolverUtility
import progressbar as pb
from datetime import date



class Tourist(object):
    tourist_count = 0

    def __init__(self, index, uid):
        self.index, self.uid = index, uid
        self.trip_df, self.time_budget = None, 0
        self.path_pdt, self.path_obs = None, None
        self.preference = None
        Tourist.tourist_count += 1


def obj_eval(s):  # enumerate all agents and calculate total errors
    alpha = list(s[:2])
    beta = [5] + list(s[2:])
    data_input = {'alpha': alpha, 'beta': beta,
                  'phi': phi,
                  'util_matrix': utility_matrix,
                  'time_matrix': edge_time_matrix,
                  'cost_matrix': edge_cost_matrix,
                  'dwell_matrix': dwell_vector,
                  'dist_matrix': edge_distance_matrix}

    # enumerate all agents
    return SolverUtility.solver_single(node_num, _agent, **data_input)


if __name__ == '__main__':
    #  Solver Database setup
    # B_star
    s = [-1.286284872, -0.286449175, 0.691566901, 0.353739632]

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

    print('Setting up solver.')


    core_process = mp.cpu_count()  # species size (each individual is our parameters here)

    """ 工事中
    # todo  等 debug完成后再加入 multi-tasking
    # n_cores = mp.cpu_count()
    # 
    # pop = chunks(agent_database, n_cores)
    # # for i in pop:
    # #     print(len(i))  # 尽可能平均
    # 
    # jobs = []
    # penalty_queue = mp.Queue()  # queue, to save results for multi_processing
    # 
    # # start process
    # 
    # for idx, chunk in enumerate(pop):
    """
    p = pb.ProgressBar(widgets=[
        ' [', pb.Timer(), '] ',
        pb.Percentage(),
        ' (', pb.ETA(), ') ',
    ])

    p.start()
    total = len(agent_database)

    H = np.zeros([len(s), len(s)])
    skipped = 0
    for _idx, _agent in enumerate(agent_database):
        p.update(int((_idx / (total - 1)) * 100))
        try:

            # res_pa = nd.Gradient(pf.eval_fun)(s)
            # res_test = pf.eval_fun(s)
            # print('The score of parameter of {}: {}'.format(s, res_test))
            # res_test = obj_eval(s)
            temp = nd.Hessian(obj_eval)(s)  # calculate Hessian for each single tourist
            H += temp  # add to Hessian for parameter
            # print('Current agent {} with prediction error {}'.format(_idx, res_test))
            pass
        except ValueError:  # 万一func value为0，那就直接跳过该tourist
            print('Skipped at agent with index {}'.format(_idx))
            skipped += 1
            continue

    res_Hessian = H/(len(agent_database) - skipped)

    p.finish()

    # 最后H要除以n
    # todo 把node, edge properties都放在main里面
    # %%
    df_hess = pd.DataFrame(res_Hessian)
    today = date.today()
    # df_grad.to_excel('Gradient_Batch.xlsx')
    df_hess.to_excel('Numerical Hessian Single {}.xlsx'.format(today.strftime("%m_%d")))

    # %% calculate standard error. Modified on Dec. 16
    variance = np.linalg.inv(res_Hessian)
    std_err = np.sqrt(np.diag(variance))

    t_value = np.array(s) / std_err
    # print second derivatives
    print('The second derivative matrix:\n {}'.format(res_Hessian))
    print('The standard error:\n {}'.format(std_err))

    print('The t values for current beta*:\n {}'.format(t_value))
