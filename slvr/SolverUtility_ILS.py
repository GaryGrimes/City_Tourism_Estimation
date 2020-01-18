""" This script contains main method for multi-tasking module, the TTDP solver using Iterated Local Search approach.
Data Wrapping is executed in parent script to avoid repetitive I/Os.
Added codes in main script to test and print route utilties. <- last update on Dec.25"""

import numpy as np
import multiprocessing as mp
from SimInfo import Solver_ILS
import datetime
import pickle
import os
import SimDataProcessing as sim_data
import pandas as pd


class Tourist(object):
    tourist_count = 0

    def __init__(self, index, uid):
        self.index, self.uid = index, uid
        self.trip_df, self.time_budget = None, 0
        self.path_pdt, self.path_obs = None, None
        self.preference = None
        Tourist.tourist_count += 1


class SolverUtility(object):
    @staticmethod
    def solver(q, process_idx, node_num, agent_database, **kwargs):  # Levenshtein distance, with path threshold filter
        # pass variables
        util_matrix, time_matrix, cost_matrix, dwell_matrix, \
        dist_matrix = kwargs['util_matrix'], kwargs['time_matrix'], \
                      kwargs['cost_matrix'], kwargs['dwell_matrix'], kwargs['dist_matrix']

        # behavioral parameters data setup
        Solver_ILS.alpha = kwargs['alpha']
        Solver_ILS.beta = kwargs['beta']
        Solver_ILS.phi = kwargs['phi']

        # save results for all agents
        _penalty, _pdt_path, _obs_path = [], [], []

        # enumerate each tourist
        # node setup

        node_properties = {'node_num': node_num,
                           'utility_matrix': util_matrix,
                           'dwell_vector': dwell_matrix}

        # edge setup

        edge_properties = {'edge_time_matrix': time_matrix,
                           'edge_cost_matrix': cost_matrix,
                           'edge_distance_matrix': dist_matrix}

        Solver_ILS.edge_setup(**edge_properties)

        iteration_size = len(agent_database)

        for _idd, _agent in enumerate(agent_database):
            if _idd > 0 and _idd % 500 == 0:
                print(
                    '--- Running optimal tours for the {} agent in {} for process {}'.format(
                        _idd, len(agent_database), mp.current_process().name))

            pref = _agent.preference
            observed_path = _agent.path_obs
            t_max, origin, destination = _agent.time_budget, observed_path[0] - 1, observed_path[-1] - 1
            visit_history = {}

            if pref is None or observed_path is None:
                continue
            # skip empty paths (no visited location)
            if len(observed_path) < 3:
                continue

            """node setup process should be here!!"""
            # ... since nodes controls visit history in path update for each agent
            Solver_ILS.node_setup(**node_properties)

            # agents setup
            agent_properties = {'time_budget': t_max,
                                'origin': origin,
                                'destination': destination,
                                'preference': pref,
                                'visited': visit_history}

            Solver_ILS.agent_setup(**agent_properties)

            # each path_op will be saved into the predicted path set for agent n
            path_pdt = []

            # %% strat up solver
            no_init_flag = 0
            # solver initialization
            initial_path = Solver_ILS.initial_solution()

            if len(initial_path) <= 2:
                no_init_flag = 1
            else:
                first_visit = initial_path[1]
                Solver_ILS.Node_list[first_visit].visit = 1

                order = initial_path
                final_order = list(order)

                # No edgeMethod in my case
                _u, _u8, _U10 = [], [], []

                counter_2 = 0
                no_improve = 0
                best_found = float('-inf')

                while no_improve < 50:
                    best_score = float('-inf')
                    local_optimum = 0

                    # print(Order)

                    while local_optimum == 0:
                        local_optimum, order, best_score = Solver_ILS.insert(order, best_score)

                    counter_2 += 1  # 2指inner loop的counter
                    v = len(order) - 1

                    _u.append(best_score)  #  U is utility memoizer
                    _u8.append(v)
                    _U10.append(max(_u))

                    if best_score > best_found:
                        best_found = best_score
                        final_order = list(order)

                        # save intermediate good paths into results
                        path_pdt.append(list(final_order))
                        no_improve = 0  # improved
                    else:
                        no_improve += 1

                    if len(order) <= 2:
                        continue
                    else:
                        s = np.random.randint(1, len(order) - 1)
                        R = np.random.randint(1, len(order) - s)  # from node S, delete R nodes (including S)

                    if s >= min(_u8):
                        s = s - min(_u8) + 1

                    order = Solver_ILS.shake(order, s, R)  # break sequence



            path_obs = list(
                np.array(_agent.path_obs) - 1)  # attraction indices in solver start from 0 (in survey start from 1)

            # last modified on Oct. 24 16:29 2019
            # last modified on Dec. 20

            if no_init_flag:
                # do compulsory fill
                path_pdt.append(Solver_ILS.comp_fill())
                pass

            """对比的是combinatorial path score"""
            selected_path = []
            if len(path_pdt) < 3:  # return directly if...
                selected_path = list(path_pdt)
            else:
                # evaluate scores for all path predicted (not penalty with the observed path here)
                # select only 10 paths with high scores for filtering
                path_pdt = path_pdt[-10:]

                path_pdt_score = []
                for _path in path_pdt:
                    # The memoizer must accept agent's preference as well.
                    path_pdt_score.append(Solver_ILS.eval_util(_path, pref))  # a list of penalties

                filter_ratio = 0.15  # predicted paths with penalties within 15% interval
                max_score = max(path_pdt_score)  # max utility score for current path

                ''' within 85% score or at least 3 paths in the predicted path set'''
                threshold = max_score - abs(filter_ratio * max_score)

                for _ in np.argsort(path_pdt_score)[::-1]:
                    if path_pdt_score[_] >= threshold:
                        selected_path.append(path_pdt[_])
                    else:
                        break

                # at least 3 paths in the set
                if len(selected_path) < 3:
                    selected_path = []
                    for _ in np.argsort(path_pdt_score)[-3:]:
                        selected_path.append(path_pdt[_])

            # -------- Path penalty evaluation --------
            # compare predicted path with observed ones
            # modified on Nov. 3 2019
            # last modified on Nov. 16

            best_path_predicted, lowest_penalty = [], float('inf')
            for _path in selected_path:
                res = Solver_ILS.path_penalty(path_obs, _path)
                if not best_path_predicted:
                    best_path_predicted, lowest_penalty = _path, res
                if res < lowest_penalty:
                    best_path_predicted, lowest_penalty = _path, res
                # print('With path score: %.2f, time: %d' % (solver.eval_util(_path, Pref).
                # solver.time_callback(_path)))
                # print('Penalty: {}'.format(res))
                # print_path(_path)

            # WRITE PREDICTED PATH AND PENALTY

            _penalty.append(lowest_penalty)
            _pdt_path.append(best_path_predicted)
            _obs_path.append(path_obs)

        sorted_indices = np.argsort(_penalty)  # according to mismatch penalty, for all tourists, from min to max
        predicted = [_pdt_path[_] for _ in sorted_indices[:10]] + [_pdt_path[_] for _ in sorted_indices[-10:]]
        observed = [_obs_path[_] for _ in sorted_indices[:10]] + [_obs_path[_] for _ in sorted_indices[-10:]]

        # unit of penalty transformed from 'meter' to 'kilometer'. Total penalty, not average.
        data = {'penalty': sum(_penalty) / 1000, 'predicted': predicted,
                'observed': observed}

        q.put((process_idx, data['penalty']))

    @staticmethod
    def solver_null(q, process_idx, node_num, agent_database, **kwargs):
        """ Using Levenshtein distance, with path threshold filter """
        # pass variables
        alpha, beta, phi, util_matrix, time_matrix, \
        cost_matrix, dwell_matrix, dist_matrix = kwargs['alpha'], kwargs['beta'], kwargs['phi'], kwargs[
            'util_matrix'], kwargs['time_matrix'], kwargs['cost_matrix'], kwargs['dwell_matrix'], kwargs['dist_matrix']

        # behavioral parameters data setup

        Solver_ILS.alpha = kwargs['alpha']
        Solver_ILS.beta = kwargs['beta']
        Solver_ILS.phi = kwargs['phi']

        # save results for all agents
        _penalty, _pdt_path, _obs_path = [], [], []

        # enumerate each tourist
        # node setup

        node_properties = {'node_num': node_num,
                           'utility_matrix': util_matrix,
                           'dwell_vector': dwell_matrix}

        # edge setup

        edge_properties = {'edge_time_matrix': time_matrix,
                           'edge_cost_matrix': cost_matrix,
                           'edge_distance_matrix': dist_matrix}

        Solver_ILS.edge_setup(**edge_properties)

        iteration_size = len(agent_database)

        for _idd, _agent in enumerate(agent_database):
            if _idd > 0 and _idd % 500 == 0:
                print(
                    '--- Running optimal tours for the {} agent in {} for process {}'.format(
                        _idd, len(agent_database), mp.current_process().name))

            pref = _agent.preference
            observed_path = _agent.path_obs
            t_max, origin, destination = _agent.time_budget, observed_path[0] - 1, observed_path[-1] - 1
            visit_history = {}

            if pref is None or observed_path is None:
                continue
            # skip empty paths (no visited location)
            if len(observed_path) < 3:
                continue

            """node setup process should be here!!"""
            # ... since nodes controls visit history in path update for each agent
            Solver_ILS.node_setup(**node_properties)

            # agents setup
            agent_properties = {'time_budget': t_max,
                                'origin': origin,
                                'destination': destination,
                                'preference': pref,
                                'visited': visit_history}

            Solver_ILS.agent_setup(**agent_properties)

            # each path_op will be saved into the predicted path set for agent n
            path_pdt = []

            # %% strat up solver
            no_init_flag = 0
            # solver initialization
            initial_path = Solver_ILS.initial_solution()

            if len(initial_path) <= 2:
                no_init_flag = 1
            else:
                first_visit = initial_path[1]
                Solver_ILS.Node_list[first_visit].visit = 1

                order = initial_path
                final_order = list(order)

                # No edgeMethod in my case
                _u, _u8, _U10 = [], [], []

                counter_2 = 0
                no_improve = 0
                best_found = float('-inf')

                while no_improve < 50:
                    best_score = float('-inf')
                    local_optimum = 0

                    # print(Order)

                    while local_optimum == 0:
                        local_optimum, order, best_score = Solver_ILS.insert(order, best_score)

                    counter_2 += 1  # 2指inner loop的counter
                    v = len(order) - 1

                    _u.append(best_score)  # TODO U is utility memo
                    _u8.append(v)
                    _U10.append(max(_u))

                    if best_score > best_found:
                        best_found = best_score
                        final_order = list(order)

                        # save intermediate good paths into results
                        path_pdt.append(list(final_order))
                        no_improve = 0  # improved
                    else:
                        no_improve += 1

                    if len(order) <= 2:
                        continue
                    else:
                        s = np.random.randint(1, len(order) - 1)
                        R = np.random.randint(1, len(order) - s)  # from node S, delete R nodes (including S)

                    if s >= min(_u8):
                        s = s - min(_u8) + 1

                    order = Solver_ILS.shake(order, s, R)

            # print('Near optimal path: {}, with total time {} min, utility {}.'.format(final_order,
            #                                                                           Solver_ILS.time_callback(
            #                                                                               final_order),
            #                                                                           Solver_ILS.eval_util(
            #                                                                               final_order)))

            # Prediction penalty evaluation. Compare the predicted paths with observed one.

            path_obs = list(
                np.array(_agent.path_obs) - 1)  # attraction indices in solver start from 0 (in survey start from 1)

            # last modified on Oct. 24 16:29 2019
            # last modified on Dec. 20

            if no_init_flag:
                # do compulsory fill
                path_pdt.append(Solver_ILS.comp_fill())
                pass

            """对比的是combinatorial path score"""
            selected_path = []
            if len(path_pdt) < 3:  # return directly if...
                selected_path = list(path_pdt)
            else:
                # evaluate scores for all path predicted (not penalty with the observed path here)
                path_pdt_score = []
                for _path in path_pdt:
                    path_pdt_score.append(Solver_ILS.eval_util(_path, pref))  # a list of penalties

                filter_ratio = 0.15  # predicted paths with penalties within 15% interval
                max_score = max(path_pdt_score)  # max utility score for current path

                ''' within 85% score or at least 3 paths in the predicted path set'''
                threshold = max_score - abs(filter_ratio * max_score)

                for _ in np.argsort(path_pdt_score)[::-1]:
                    if path_pdt_score[_] >= threshold:
                        selected_path.append(path_pdt[_])
                    else:
                        break
                # at least 3 paths in the set
                if len(selected_path) < 3:
                    selected_path = []
                    for _ in np.argsort(path_pdt_score)[-3:]:
                        selected_path.append(path_pdt[_])

            # -------- Path penalty evaluation --------
            # compare predicted path with observed ones
            # modified on Nov. 3 2019
            # last modified on Nov. 16

            best_path_predicted, lowest_penalty = [], float('inf')
            for _path in selected_path:
                res = Solver_ILS.path_penalty(path_obs, _path)
                if not best_path_predicted:
                    best_path_predicted, lowest_penalty = _path, res
                if res < lowest_penalty:
                    best_path_predicted, lowest_penalty = _path, res
                # print('With path score: %.2f, time: %d' % (solver.eval_util(_path, Pref).
                # solver.time_callback(_path)))
                # print('Penalty: {}'.format(res))
                # print_path(_path)

            # WRITE PREDICTED PATH AND PENALTY

            _penalty.append(lowest_penalty)
            _pdt_path.append(best_path_predicted)
            _obs_path.append(path_obs)

        sorted_indices = np.argsort(_penalty)  # according to mismatch penalty, for all tourists, from min to max
        predicted = [_pdt_path[_] for _ in sorted_indices[:10]] + [_pdt_path[_] for _ in sorted_indices[-10:]]
        observed = [_obs_path[_] for _ in sorted_indices[:10]] + [_obs_path[_] for _ in sorted_indices[-10:]]

        # unit of penalty transformed from 'meter' to 'kilometer'. Total penalty, not average.
        data = {'penalty': sum(_penalty) / 1000, 'predicted': predicted,
                'observed': observed}

        q.put((process_idx, data['penalty']))

    @staticmethod
    def solver_trip_stat(q, process_idx, node_num, agent_database, **kwargs):
        """ Predict tours and generate trip frequency table for a single set of parameters. """

        # Levenshtein distance, with path threshold filter

        # pass variables
        alpha, beta, phi, util_matrix, time_matrix, \
        cost_matrix, dwell_matrix, dist_matrix = kwargs['alpha'], kwargs['beta'], kwargs['phi'], kwargs[
            'util_matrix'], kwargs['time_matrix'], kwargs['cost_matrix'], kwargs['dwell_matrix'], kwargs['dist_matrix']

        # behavioral parameters data setup

        Solver_ILS.alpha = kwargs['alpha']
        Solver_ILS.beta = kwargs['beta']
        Solver_ILS.phi = kwargs['phi']

        # save results for all agents
        _penalty, _pdt_path, _obs_path = [], [], []
        #  trip_table 变量， 并在sent to queue的时候折叠为array (using .flatten()), customer side use reshape([37, 37])
        trip_table = np.zeros((37, 37), dtype=int)

        # enumerate each tourist
        # node setup

        node_properties = {'node_num': node_num,
                           'utility_matrix': util_matrix,
                           'dwell_vector': dwell_matrix}

        # edge setup

        edge_properties = {'edge_time_matrix': time_matrix,
                           'edge_cost_matrix': cost_matrix,
                           'edge_distance_matrix': dist_matrix}

        Solver_ILS.edge_setup(**edge_properties)

        iteration_size = len(agent_database)

        for _idd, _agent in enumerate(agent_database):
            if _idd > 0 and _idd % 500 == 0:
                print(
                    '--- Running optimal tours for the {} agent in {} for process {}'.format(
                        _idd, len(agent_database), mp.current_process().name))

            pref = _agent.preference
            observed_path = _agent.path_obs
            t_max, origin, destination = _agent.time_budget, observed_path[0] - 1, observed_path[-1] - 1
            visit_history = {}

            if pref is None or observed_path is None:
                continue
            # skip empty paths (no visited location)
            if len(observed_path) < 3:
                continue

            """node setup process should be here!!"""
            # ... since nodes controls visit history in path update for each agent
            Solver_ILS.node_setup(**node_properties)

            # agents setup
            agent_properties = {'time_budget': t_max,
                                'origin': origin,
                                'destination': destination,
                                'preference': pref,
                                'visited': visit_history}

            Solver_ILS.agent_setup(**agent_properties)

            # each path_op will be saved into the predicted path set for agent n
            path_pdt = []

            # %% strat up solver
            no_init_flag = 0
            # solver initialization
            initial_path = Solver_ILS.initial_solution()

            if len(initial_path) <= 2:
                no_init_flag = 1
            else:
                first_visit = initial_path[1]
                Solver_ILS.Node_list[first_visit].visit = 1

                order = initial_path
                final_order = list(order)

                # No edgeMethod in my case
                _u, _u8, _U10 = [], [], []

                counter_2 = 0
                no_improve = 0
                best_found = float('-inf')

                while no_improve < 50:
                    best_score = float('-inf')
                    local_optimum = 0

                    # print(Order)

                    while local_optimum == 0:
                        local_optimum, order, best_score = Solver_ILS.insert(order, best_score)

                    counter_2 += 1  # 2指inner loop的counter
                    v = len(order) - 1

                    _u.append(best_score)  # TODO U is utility memo
                    _u8.append(v)
                    _U10.append(max(_u))

                    if best_score > best_found:
                        best_found = best_score
                        final_order = list(order)

                        # save intermediate good paths into results
                        path_pdt.append(list(final_order))
                        no_improve = 0  # improved
                    else:
                        no_improve += 1

                    if len(order) <= 2:
                        continue
                    else:
                        s = np.random.randint(1, len(order) - 1)
                        R = np.random.randint(1, len(order) - s)  # from node S, delete R nodes (including S)

                    if s >= min(_u8):
                        s = s - min(_u8) + 1

                    order = Solver_ILS.shake(order, s, R)

            # print('Near optimal path: {}, with total time {} min, utility {}.'.format(final_order,
            #                                                                           Solver_ILS.time_callback(
            #                                                                               final_order),
            #                                                                           Solver_ILS.eval_util(
            #                                                                               final_order)))

            # Prediction penalty evaluation. Compare the predicted paths with observed one.

            path_obs = list(
                np.array(_agent.path_obs) - 1)  # attraction indices in solver start from 0 (in survey start from 1)

            # last modified on Oct. 24 16:29 2019
            # last modified on Dec. 20

            if no_init_flag:
                # do compulsory fill
                path_pdt.append(Solver_ILS.comp_fill())
                pass

            """对比的是combinatorial path score"""
            selected_path = []
            if len(path_pdt) < 3:  # return directly if...
                selected_path = list(path_pdt)
            else:
                # evaluate scores for all path predicted (not penalty with the observed path here)
                path_pdt_score = []
                for _path in path_pdt:
                    path_pdt_score.append(Solver_ILS.eval_util(_path))  # a list of penalties

                filter_ratio = 0.15  # predicted paths with penalties within 15% interval
                max_score = max(path_pdt_score)  # max utility score for current path

                ''' within 85% score or at least 3 paths in the predicted path set'''
                threshold = max_score - abs(filter_ratio * max_score)

                for _ in np.argsort(path_pdt_score)[::-1]:
                    if path_pdt_score[_] >= threshold:
                        selected_path.append(path_pdt[_])
                    else:
                        break
                # at least 3 paths in the set
                if len(selected_path) < 3:
                    selected_path = []
                    for _ in np.argsort(path_pdt_score)[-3:]:
                        selected_path.append(path_pdt[_])

            # -------- Path penalty evaluation --------
            # compare predicted path with observed ones
            # modified on Nov. 3 2019
            # last modified on Nov. 16

            best_path_predicted, lowest_penalty = [], float('inf')
            for _path in selected_path:
                res = Solver_ILS.path_penalty(path_obs, _path)
                if not best_path_predicted:
                    best_path_predicted, lowest_penalty = _path, res
                if res < lowest_penalty:
                    best_path_predicted, lowest_penalty = _path, res
                # print('With path score: %.2f, time: %d' % (solver.eval_util(_path, Pref).
                # solver.time_callback(_path)))
                # print('Penalty: {}'.format(res))
                # print_path(_path)

            # WRITE PREDICTED PATH AND PENALTY

            _penalty.append(lowest_penalty)
            _pdt_path.append(best_path_predicted)
            _obs_path.append(path_obs)

            # parse trip frequency
            for _idx in range(len(best_path_predicted) - 1):
                _o, _d = best_path_predicted[_idx], best_path_predicted[_idx + 1]
                try:
                    trip_table[_o, _d] += 1
                except IndexError:
                    continue

        # if oversize (capacity limit) encountered, try dump the results by pickle and read again
        name = 'predicted_trip_table_{}.pickle'.format(process_idx)
        file = open(os.path.join(os.path.dirname(__file__), 'SimInfo', 'temp', name), 'wb')
        pickle.dump(trip_table, file)
        q.put(process_idx)

    @staticmethod
    def solver_util_scatter(q, process_idx, node_num, agent_database, **kwargs):
        """ Predict tours and generate tuples with observed and predicted trip
        utilities for a single set of parameters. Using Levenshtein distance, with path threshold filter."""

        # pass variables
        alpha, beta, phi, util_matrix, time_matrix, \
        cost_matrix, dwell_matrix, dist_matrix = kwargs['alpha'], kwargs['beta'], kwargs['phi'], kwargs[
            'util_matrix'], kwargs['time_matrix'], kwargs['cost_matrix'], kwargs['dwell_matrix'], kwargs['dist_matrix']

        # behavioral parameters data setup

        Solver_ILS.alpha = kwargs['alpha']
        Solver_ILS.beta = kwargs['beta']
        Solver_ILS.phi = kwargs['phi']

        # save results for all agents
        _penalty, _pdt_path, _obs_path = [], [], []

        util_scatter_dots = []

        # enumerate each tourist

        # node setup
        node_properties = {'node_num': node_num,
                           'utility_matrix': util_matrix,
                           'dwell_vector': dwell_matrix}

        # edge setup
        edge_properties = {'edge_time_matrix': time_matrix,
                           'edge_cost_matrix': cost_matrix,
                           'edge_distance_matrix': dist_matrix}

        Solver_ILS.edge_setup(**edge_properties)

        for _idd, _agent in enumerate(agent_database):
            if _idd > 0 and _idd % 500 == 0:
                print(
                    '--- Running optimal tours for the {} agent in {} for process {}'.format(
                        _idd, len(agent_database), mp.current_process().name))

            pref = _agent.preference
            observed_path = _agent.path_obs
            t_max, origin, destination = _agent.time_budget, observed_path[0] - 1, observed_path[-1] - 1
            visit_history = {}

            if pref is None or observed_path is None:
                continue
            # skip empty paths (no visited location)
            if len(observed_path) < 3:
                continue

            """node setup process should be here!!"""
            # ... since nodes controls visit history in path update for each agent
            Solver_ILS.node_setup(**node_properties)

            # agents setup
            agent_properties = {'time_budget': t_max,
                                'origin': origin,
                                'destination': destination,
                                'preference': pref,
                                'visited': visit_history}

            Solver_ILS.agent_setup(**agent_properties)

            # each path_op will be saved into the predicted path set for agent n
            path_pdt = []

            # %% strat up solver
            no_init_flag = 0
            # solver initialization
            initial_path = Solver_ILS.initial_solution()

            if len(initial_path) <= 2:
                no_init_flag = 1
            else:
                first_visit = initial_path[1]
                Solver_ILS.Node_list[first_visit].visit = 1

                order = initial_path
                final_order = list(order)

                # No edgeMethod in my case
                _u, _u8, _U10 = [], [], []

                counter_2 = 0
                no_improve = 0
                best_found = float('-inf')

                while no_improve < 50:
                    best_score = float('-inf')
                    local_optimum = 0

                    # print(Order)

                    while local_optimum == 0:
                        local_optimum, order, best_score = Solver_ILS.insert(order, best_score)

                    counter_2 += 1  # 2指inner loop的counter
                    v = len(order) - 1

                    _u.append(best_score)  # TODO U is utility memo
                    _u8.append(v)
                    _U10.append(max(_u))

                    if best_score > best_found:
                        best_found = best_score
                        final_order = list(order)

                        # save intermediate good paths into results
                        path_pdt.append(list(final_order))
                        no_improve = 0  # improved
                    else:
                        no_improve += 1

                    if len(order) <= 2:
                        continue
                    else:
                        s = np.random.randint(1, len(order) - 1)
                        R = np.random.randint(1, len(order) - s)  # from node S, delete R nodes (including S)

                    if s >= min(_u8):
                        s = s - min(_u8) + 1

                    order = Solver_ILS.shake(order, s, R)

            # Prediction penalty evaluation. Compare the predicted paths with observed one.

            path_obs = list(
                np.array(_agent.path_obs) - 1)  # attraction indices in solver start from 0 (in survey start from 1)

            # last modified on Oct. 24 16:29 2019
            # last modified on Dec. 20

            if no_init_flag:
                # do compulsory fill
                path_pdt.append(Solver_ILS.comp_fill())
                pass

            """ Path score = experienced utiltiy. Not total penalty for a set of parameters"""

            selected_path, selected_util = [], []
            if len(path_pdt) < 3:  # return directly if...
                selected_path = list(path_pdt)
                for _path in selected_path:
                    selected_util.append(Solver_ILS.eval_util(_path))
            else:
                # evaluate scores for all path predicted (not penalty with the observed path here)
                path_pdt_score = []
                for _path in path_pdt:
                    path_pdt_score.append(Solver_ILS.eval_util(_path))  # a list of penalties

                filter_ratio = 0.15  # predicted paths with penalties within 15% interval
                max_score = max(path_pdt_score)  # max utility score for current path

                ''' within 85% score or at least 3 paths in the predicted path set'''
                threshold = max_score - abs(filter_ratio * max_score)

                for _ in np.argsort(path_pdt_score)[::-1]:
                    if path_pdt_score[_] >= threshold:
                        selected_path.append(path_pdt[_])
                        selected_util.append(path_pdt_score[_])
                    else:
                        break

                # at least 3 paths in the set
                if len(selected_path) < 3:
                    selected_path, selected_util = [], []
                    for _ in np.argsort(path_pdt_score)[-3:]:
                        selected_path.append(path_pdt[_])
                        selected_util.append(path_pdt_score[_])

            # -------- Path penalty evaluation --------
            """ compares the observed utility with predicted """
            # compare predicted path with observed
            path_obs_util = Solver_ILS.eval_util(path_obs)

            # 一条和observed有最相近utility的predcited path，并generate utility tuple (predicted and observed)
            res = [abs(path_obs_util - selected_util[_]) for _ in range(len(selected_util))]
            # the one with smallest difference
            best_path_predicted, lowest_penalty = selected_path[np.argsort(res)[0]], np.sort(res)[0]

            util_tuple = (path_obs_util, selected_util[np.argsort(res)[0]])
            util_scatter_dots.append(util_tuple)

            # WRITE PREDICTED PATH AND PENALTY

            _penalty.append(lowest_penalty)
            _pdt_path.append(best_path_predicted)
            _obs_path.append(path_obs)

        # if oversize (capacity limit) encountered, try dump the results by pickle and read again
        name = 'utility_tuples_{}.pickle'.format(process_idx)
        file = open(os.path.join(os.path.dirname(__file__), 'SimInfo', 'temp', 'scatter plot', name), 'wb')
        pickle.dump(util_scatter_dots, file)  # dump file

        q.put(process_idx)

    @staticmethod
    def solver_LD_noPF(q, process_idx, node_num, agent_database, **kwargs):
        """ with levenshtein distance, no path threshold filter"""
        # pass variables
        alpha, beta, phi, util_matrix, time_matrix, cost_matrix, dwell_matrix, dist_matrix = kwargs['alpha'], \
                                                                                             kwargs['beta'], kwargs[
                                                                                                 'phi'], kwargs[
                                                                                                 'util_matrix'], kwargs[
                                                                                                 'time_matrix'], kwargs[
                                                                                                 'cost_matrix'], kwargs[
                                                                                                 'dwell_matrix'], \
                                                                                             kwargs[
                                                                                                 'dist_matrix']

        # behavioral parameters data setup

        Solver_ILS.alpha = kwargs['alpha']
        Solver_ILS.beta = kwargs['beta']
        Solver_ILS.phi = kwargs['phi']

        # save results for all agents
        _penalty, _pdt_path, _obs_path = [], [], []

        # enumerate each tourist
        # node setup

        node_properties = {'node_num': node_num,
                           'utility_matrix': util_matrix,
                           'dwell_vector': dwell_matrix}

        # edge setup

        edge_properties = {'edge_time_matrix': time_matrix,
                           'edge_cost_matrix': cost_matrix,
                           'edge_distance_matrix': dist_matrix}

        Solver_ILS.edge_setup(**edge_properties)

        iteration_size = len(agent_database)

        for _idd, _agent in enumerate(agent_database):
            if _idd > 0 and _idd % 500 == 0:
                print(
                    '--- Running optimal tours for the {} agent in {} for process {}'.format(
                        _idd, len(agent_database), mp.current_process().name))

            pref = _agent.preference
            observed_path = _agent.path_obs
            t_max, origin, destination = _agent.time_budget, observed_path[0] - 1, observed_path[-1] - 1

            # todo
            """ 如果要考虑multiple-day travel,则可以加一个for _ in range(_agent.day_of_travel)，
            每一天的predicted path加入visit_history"""
            visit_history = {}

            if pref is None or observed_path is None:
                continue
            # skip empty paths (no visited location)
            if len(observed_path) < 3:
                continue

            """node setup process should be here!!"""
            # ... since nodes controls visit history in path update for each agent
            Solver_ILS.node_setup(**node_properties)

            # agents setup
            agent_properties = {'time_budget': t_max,
                                'origin': origin,
                                'destination': destination,
                                'preference': pref,
                                'visited': visit_history}

            Solver_ILS.agent_setup(**agent_properties)

            # each path_op will be saved into the predicted path set for agent n
            path_pdt = []

            # %% strat up solver
            # solver initialization
            initial_path = Solver_ILS.initial_solution()

            # skip agents with empty initialized path
            if not initial_path:
                continue

            if len(initial_path) <= 2:
                final_order = initial_path
            else:
                first_visit = initial_path[1]
                Solver_ILS.Node_list[first_visit].visit = 1

                order = initial_path
                final_order = list(order)

                # No edgeMethod in my case
                _u, _u8, _U10 = [], [], []

                counter_2 = 0
                no_improve = 0
                best_found = float('-inf')

                while no_improve < 50:
                    best_score = float('-inf')
                    local_optimum = 0

                    # print(Order)

                    while local_optimum == 0:
                        local_optimum, order, best_score = Solver_ILS.insert(order, best_score)

                    counter_2 += 1  # 2指inner loop的counter
                    v = len(order) - 1

                    _u.append(best_score)  # TODO U is utility memo
                    _u8.append(v)
                    _U10.append(max(_u))

                    if best_score > best_found:
                        best_found = best_score
                        final_order = list(order)

                        # save intermediate good paths into results
                        path_pdt.append(list(final_order))
                        no_improve = 0  # improved
                    else:
                        no_improve += 1

                    if len(order) <= 2:
                        continue
                    else:
                        s = np.random.randint(1, len(order) - 1)
                        R = np.random.randint(1, len(order) - s)  # from node S, delete R nodes (including S)

                    if s >= min(_u8):
                        s = s - min(_u8) + 1

                    order = Solver_ILS.shake(order, s, R)

            # print('Near optimal path: {}, with total time {} min, utility {}.'.format(final_order,
            #                                                                           Solver_ILS.time_callback(
            #                                                                               final_order),
            #                                                                           Solver_ILS.eval_util(
            #                                                                               final_order)))

            # Prediction penalty evaluation. Compare the predicted paths with observed one.

            path_obs = list(
                np.array(_agent.path_obs) - 1)  # attraction indices in solver start from 0 (in survey start from 1)

            # last modified on Oct. 24 16:29 2019

            """对比的是combinatorial path score"""

            # -------- Path penalty evaluation --------
            # compare predicted path with observed ones
            # modified on Nov. 3 2019
            # last modified on Nov. 16

            best_path_predicted, lowest_penalty = [], float('inf')
            for _path in path_pdt:
                res = Solver_ILS.path_penalty(path_obs, _path)  # path penalty (prediction error)
                if not best_path_predicted:
                    best_path_predicted, lowest_penalty = _path, res
                if res < lowest_penalty:
                    best_path_predicted, lowest_penalty = _path, res
                # print('With path score: %.2f, time: %d' % (solver.eval_util(_path, Pref).
                # solver.time_callback(_path)))
                # print('Penalty: {}'.format(res))
                # print_path(_path)

            # WRITE PREDICTED PATH AND PENALTY

            _penalty.append(lowest_penalty)
            _pdt_path.append(best_path_predicted)
            _obs_path.append(path_obs)

            # update progress bar

        # todo 加上现在的lowest_penalty，可以先对penalty进行argsort然后取index
        # TODO create a dict? tuple (idx, data) . data includes penalty,
        #  [top 10 pdt_paths, least 10 pdt_paths], [top 10 obs_paths, least 10 obs_paths]

        sorted_indices = np.argsort(_penalty)  # according to mismatch penalty, for all tourists, from min to max
        predicted = [_pdt_path[_] for _ in sorted_indices[:10]] + [_pdt_path[_] for _ in sorted_indices[-10:]]
        observed = [_obs_path[_] for _ in sorted_indices[:10]] + [_obs_path[_] for _ in sorted_indices[-10:]]

        data = {'penalty': sum(_penalty) / 1000, 'predicted': predicted,
                'observed': observed}  # unit of penalty transformed from 'meter' to 'kilometer'.

        q.put((process_idx, data['penalty']))

    @staticmethod  # levenshtein distance with path threshold filter
    def solver_single(node_num, agent, **kwargs):  # levestain distance, with path threshold filter
        '''the solver function is for single agent'''

        # pass variables
        alpha, beta, phi, util_matrix, time_matrix, cost_matrix, dwell_matrix, dist_matrix = kwargs['alpha'], \
                                                                                             kwargs['beta'], kwargs[
                                                                                                 'phi'], kwargs[
                                                                                                 'util_matrix'], kwargs[
                                                                                                 'time_matrix'], kwargs[
                                                                                                 'cost_matrix'], kwargs[
                                                                                                 'dwell_matrix'], \
                                                                                             kwargs[
                                                                                                 'dist_matrix']

        # behavioral parameters data setup

        Solver_ILS.alpha = kwargs['alpha']
        Solver_ILS.beta = kwargs['beta']
        Solver_ILS.phi = kwargs['phi']

        # node setup

        node_properties = {'node_num': node_num,
                           'utility_matrix': util_matrix,
                           'dwell_vector': dwell_matrix}

        # edge setup

        edge_properties = {'edge_time_matrix': time_matrix,
                           'edge_cost_matrix': cost_matrix,
                           'edge_distance_matrix': dist_matrix}

        Solver_ILS.edge_setup(**edge_properties)

        pref = agent.preference
        observed_path = agent.path_obs
        t_max, origin, destination = agent.time_budget, observed_path[0] - 1, observed_path[-1] - 1

        visit_history = {}

        if pref is None or observed_path is None:
            raise ValueError('This agent cannot be evaluated')
        # skip empty paths (no visited location)
        if len(observed_path) < 3:
            raise ValueError('This agent cannot be evaluated')

        # ... since nodes controls visit history in path update for each agent
        Solver_ILS.node_setup(**node_properties)

        # agents setup
        agent_properties = {'time_budget': t_max,
                            'origin': origin,
                            'destination': destination,
                            'preference': pref,
                            'visited': visit_history}

        Solver_ILS.agent_setup(**agent_properties)

        # each path_op will be saved into the predicted path set for agent n
        path_pdt = []

        # %% strat up solver
        # solver initialization
        initial_path = Solver_ILS.initial_solution()

        # skip agents with empty initialized path
        if not initial_path:
            raise ValueError('This agent cannot be evaluated')

        if len(initial_path) <= 2:
            final_order = initial_path
        else:
            first_visit = initial_path[1]
            Solver_ILS.Node_list[first_visit].visit = 1

            order = initial_path
            final_order = list(order)

            # No edgeMethod in my case
            _u, _u8, _U10 = [], [], []

            counter_2 = 0
            no_improve = 0
            best_found = float('-inf')

            while no_improve < 50:
                best_score = float('-inf')
                local_optimum = 0

                # print(Order)

                while local_optimum == 0:
                    local_optimum, order, best_score = Solver_ILS.insert(order, best_score)

                counter_2 += 1  # 2指inner loop的counter
                v = len(order) - 1

                _u.append(best_score)  # TODO U is utility memo
                _u8.append(v)
                _U10.append(max(_u))

                if best_score > best_found:
                    best_found = best_score
                    final_order = list(order)

                    # save intermediate good paths into results
                    path_pdt.append(list(final_order))
                    no_improve = 0  # improved
                else:
                    no_improve += 1

                if len(order) <= 2:
                    continue
                else:
                    s = np.random.randint(1, len(order) - 1)
                    R = np.random.randint(1, len(order) - s)  # from node S, delete R nodes (including S)

                if s >= min(_u8):
                    s = s - min(_u8) + 1

                order = Solver_ILS.shake(order, s, R)

        # print('Near optimal path: {}, with total time {} min, utility {}.'.format(final_order,
        #                                                                           Solver_ILS.time_callback(
        #                                                                               final_order),
        #                                                                           Solver_ILS.eval_util(
        #                                                                               final_order)))

        # Prediction penalty evaluation. Compare the predicted paths with observed one.

        path_obs = list(
            np.array(agent.path_obs) - 1)  # attraction indices in solver start from 0 (in survey start from 1)

        # last modified on Oct. 24 16:29 2019

        """对比的是combinatorial path score"""
        selected_path = []
        if len(path_pdt) < 3:  # return directly if...
            selected_path = list(path_pdt)
        else:
            # evaluate scores for all path predicted (not penalty with the observed path here)
            path_pdt_score = []
            for _path in path_pdt:
                path_pdt_score.append(Solver_ILS.eval_util(_path))  # a list of penalties

            filter_ratio = 0.15  # predicted paths with penalties within 15% interval
            max_score = max(path_pdt_score)  # max utility score for current path

            ''' within 90% score or at least 3 paths in the predicted path set'''
            threshold = max_score - abs(filter_ratio * max_score)

            for _ in np.argsort(path_pdt_score)[::-1]:
                if path_pdt_score[_] >= threshold:
                    selected_path.append(path_pdt[_])
                else:
                    break
            # at least 3 paths in the set
            if len(selected_path) < 3:
                selected_path = []
                for _ in np.argsort(path_pdt_score)[-3:]:
                    selected_path.append(path_pdt[_])

        # -------- Path penalty evaluation --------
        # compare predicted path with observed ones
        # modified on Nov. 3 2019
        # last modified on Nov. 16

        best_path_predicted, lowest_penalty = [], float('inf')
        for _path in selected_path:
            res = Solver_ILS.path_penalty(path_obs, _path)
            if not best_path_predicted:
                best_path_predicted, lowest_penalty = _path, res
            if res < lowest_penalty:
                best_path_predicted, lowest_penalty = _path, res

        return lowest_penalty  # unit in m

    @staticmethod
    def solver_debug(process_idx, node_num, agent_database, **kwargs):  #
        # pass variables
        alpha, beta, phi, util_matrix, time_matrix, cost_matrix, dwell_matrix, dist_matrix = kwargs['alpha'], \
                                                                                             kwargs['beta'], kwargs[
                                                                                                 'phi'], kwargs[
                                                                                                 'util_matrix'], kwargs[
                                                                                                 'time_matrix'], kwargs[
                                                                                                 'cost_matrix'], kwargs[
                                                                                                 'dwell_matrix'], \
                                                                                             kwargs[
                                                                                                 'dist_matrix']

        # behavioral parameters data setup

        Solver_ILS.alpha = kwargs['alpha']
        Solver_ILS.beta = kwargs['beta']
        Solver_ILS.phi = kwargs['phi']

        # save results for all agents
        _penalty, _pdt_path, _obs_path = [], [], []

        # enumerate each tourist
        # node setup

        node_properties = {'node_num': node_num,
                           'utility_matrix': util_matrix,
                           'dwell_vector': dwell_matrix}

        # edge setup

        edge_properties = {'edge_time_matrix': time_matrix,
                           'edge_cost_matrix': cost_matrix,
                           'edge_distance_matrix': dist_matrix}

        Solver_ILS.edge_setup(**edge_properties)

        iteration_size = len(agent_database)

        # enumerate all tourists
        success_usr_cnt = 0
        success_set = set()
        initial_skip = 0

        # error type for unsuccessful tourists
        err_emty_info = []
        err_no_path = []
        err_init = []

        for _idd, _agent in enumerate(agent_database):
            if _idd > 0 and _idd % 500 == 0:
                print(
                    '--- Running optimal tours for the {} agent in {} for process {}'.format(
                        _idd, len(agent_database), mp.current_process().name))

            pref = _agent.preference
            observed_path = _agent.path_obs
            t_max, origin, destination = _agent.time_budget, observed_path[0] - 1, observed_path[-1] - 1

            # todo
            """ 如果要考虑multiple-day travel,则可以加一个for _ in range(_agent.day_of_travel)，
            每一天的predicted path加入visit_history"""
            visit_history = {}

            if pref is None or observed_path is None:
                err_emty_info.append(_idd)  # for debug
                continue
            # skip empty paths (no visited location)
            if len(observed_path) < 3:
                err_no_path.append(_idd)  # for debug
                continue

            start_time = datetime.datetime.now()

            """node setup process should be here!!"""
            # ... since nodes controls visit history in path update for each agent
            Solver_ILS.node_setup(**node_properties)

            # agents setup
            agent_properties = {'time_budget': t_max,
                                'origin': origin,
                                'destination': destination,
                                'preference': pref,
                                'visited': visit_history}

            Solver_ILS.agent_setup(**agent_properties)

            # each path_op will be saved into the predicted path set for agent n
            path_pdt = []

            # %% strat up solver
            no_init_flag = 0
            # solver initialization
            initial_path = Solver_ILS.initial_solution()

            # skip agents with empty initialized path
            if not initial_path:
                err_init.append(_idd)
                # initial_skip += 1
                # continue

            if len(initial_path) <= 2:
                no_init_flag = 1
            else:
                first_visit = initial_path[1]
                Solver_ILS.Node_list[first_visit].visit = 1

                order = initial_path
                final_order = list(order)

                # No edgeMethod in my case
                _u, _u8, _U10 = [], [], []

                counter_2 = 0
                no_improve = 0
                best_found = float('-inf')

                while no_improve < 50:
                    best_score = float('-inf')
                    local_optimum = 0

                    # print(Order)

                    while local_optimum == 0:
                        local_optimum, order, best_score = Solver_ILS.insert(order, best_score)

                    counter_2 += 1  # 2指inner loop的counter
                    v = len(order) - 1

                    _u.append(best_score)  # TODO U is utility memo
                    _u8.append(v)
                    _U10.append(max(_u))

                    if best_score > best_found:
                        best_found = best_score
                        final_order = list(order)

                        # save intermediate good paths into results
                        path_pdt.append(list(final_order))
                        no_improve = 0  # improved
                    else:
                        no_improve += 1

                    if len(order) <= 2:
                        continue
                    else:
                        s = np.random.randint(1, len(order) - 1)
                        R = np.random.randint(1, len(order) - s)  # from node S, delete R nodes (including S)

                    if s >= min(_u8):
                        s = s - min(_u8) + 1

                    order = Solver_ILS.shake(order, s, R)

            # print('Near optimal path: {}, with total time {} min, utility {}.'.format(final_order,
            #                                                                           Solver_ILS.time_callback(
            #                                                                               final_order),
            #                                                                           Solver_ILS.eval_util(
            #                                                                               final_order)))

            # Prediction penalty evaluation. Compare the predicted paths with observed one.

            path_obs = list(
                np.array(_agent.path_obs) - 1)  # attraction indices in solver start from 0 (in survey start from 1)

            # last modified on Oct. 24 16:29 2019
            # last modified on Dec. 20

            if no_init_flag:
                # do compulsory fill
                path_pdt.append(Solver_ILS.comp_fill())
                pass

            """对比的是combinatorial path score"""
            selected_path = []
            if len(path_pdt) < 3:  # return directly if...
                selected_path = list(path_pdt)
            else:
                # evaluate scores for all path predicted (not penalty with the observed path here)
                path_pdt_score = []
                for _path in path_pdt:
                    path_pdt_score.append(Solver_ILS.eval_util(_path))  # a list of penalties

                filter_ratio = 0.15  # predicted paths with penalties within 15% interval
                max_score = max(path_pdt_score)  # max utility score for current path

                ''' within 90% score or at least 3 paths in the predicted path set'''
                threshold = max_score - abs(filter_ratio * max_score)

                for _ in np.argsort(path_pdt_score)[::-1]:
                    if path_pdt_score[_] >= threshold:
                        selected_path.append(path_pdt[_])
                    else:
                        break
                # at least 3 paths in the set
                if len(selected_path) < 3:
                    selected_path = []
                    for _ in np.argsort(path_pdt_score)[-3:]:
                        selected_path.append(path_pdt[_])

            # -------- Path penalty evaluation --------
            # compare predicted path with observed ones
            # modified on Nov. 3 2019
            # last modified on Nov. 16

            best_path_predicted, lowest_penalty = [], float('inf')
            for _path in selected_path:
                res = Solver_ILS.path_penalty(path_obs, _path)
                if not best_path_predicted:
                    best_path_predicted, lowest_penalty = _path, res
                if res < lowest_penalty:
                    best_path_predicted, lowest_penalty = _path, res
                # print('With path score: %.2f, time: %d' % (solver.eval_util(_path, Pref).
                # solver.time_callback(_path)))
                # print('Penalty: {}'.format(res))
                # print_path(_path)

            # WRITE PREDICTED PATH AND PENALTY

            _penalty.append(lowest_penalty)
            _pdt_path.append(best_path_predicted)
            _obs_path.append(path_obs)

            end_time = datetime.datetime.now()
            success_usr_cnt += 1
            success_set.add(_idd)
            t_passed = (end_time - start_time).seconds
            if t_passed > 60:
                print('------ Evaluation time: {}s for agent id {}------\n'.format(t_passed, _idd))
            # update progress bar

        # todo 加上现在的lowest_penalty，可以先对penalty进行argsort然后取index
        # TODO create a dict? tuple (idx, data) . data includes penalty,
        #  [top 10 pdt_paths, least 10 pdt_paths], [top 10 obs_paths, least 10 obs_paths]

        sorted_indices = np.argsort(_penalty)  # according to mismatch penalty, for all tourists, from min to max
        predicted = [_pdt_path[_] for _ in sorted_indices[:10]] + [_pdt_path[_] for _ in sorted_indices[-10:]]
        observed = [_obs_path[_] for _ in sorted_indices[:10]] + [_obs_path[_] for _ in sorted_indices[-10:]]

        data = {'penalty': sum(_penalty) / 1000, 'predicted': predicted,
                'observed': observed}  # unit of penalty transformed from 'meter' to 'kilometer'.

        print('Successfully evaluated {} users. {} users were skipped, including {} of initial path.'.format(
            success_usr_cnt,
            len(agent_database) - success_usr_cnt, initial_skip))

        res_dict = {'process': process_idx,
                    'penalty': sum(_penalty) / 1000,
                    'initial skip': initial_skip,
                    'error_emty_info': err_emty_info,
                    'error_init': err_init,
                    'error_no_path': err_no_path}
        return res_dict  # for debug

    @staticmethod  # levestain distance, no path threshold filter
    def solver_debug_mp(q, process_idx, node_num, agent_database, **kwargs):  #
        # pass variables
        alpha, beta, phi, util_matrix, time_matrix, cost_matrix, dwell_matrix, dist_matrix = kwargs['alpha'], \
                                                                                             kwargs['beta'], kwargs[
                                                                                                 'phi'], kwargs[
                                                                                                 'util_matrix'], kwargs[
                                                                                                 'time_matrix'], kwargs[
                                                                                                 'cost_matrix'], kwargs[
                                                                                                 'dwell_matrix'], \
                                                                                             kwargs[
                                                                                                 'dist_matrix']

        # behavioral parameters data setup

        Solver_ILS.alpha = kwargs['alpha']
        Solver_ILS.beta = kwargs['beta']
        Solver_ILS.phi = kwargs['phi']

        # save results for all agents
        _penalty, _pdt_path, _obs_path = [], [], []

        # enumerate each tourist
        # node setup

        node_properties = {'node_num': node_num,
                           'utility_matrix': util_matrix,
                           'dwell_vector': dwell_matrix}

        # edge setup

        edge_properties = {'edge_time_matrix': time_matrix,
                           'edge_cost_matrix': cost_matrix,
                           'edge_distance_matrix': dist_matrix}

        Solver_ILS.edge_setup(**edge_properties)

        iteration_size = len(agent_database)

        # enumerate all tourists
        success_usr_cnt = 0
        success_set = set()

        # error type for unsuccessful tourists
        err_emty_info = []
        err_no_path = []
        err_init = []

        for _idd, _agent in enumerate(agent_database):
            if _idd > 0 and _idd % 500 == 0:
                print(
                    '--- Running optimal tours for the {} agent in {} for process {}'.format(
                        _idd, len(agent_database), mp.current_process().name))

            pref = _agent.preference
            observed_path = _agent.path_obs
            t_max, origin, destination = _agent.time_budget, observed_path[0] - 1, observed_path[-1] - 1

            visit_history = {}

            if pref is None or observed_path is None:
                err_emty_info.append(_idd)  # for debug
                continue
            # skip empty paths (no visited location)
            if len(observed_path) < 3:
                err_no_path.append(_idd)  # for debug
                continue

            start_time = datetime.datetime.now()

            """node setup process should be here!!"""
            # ... since nodes controls visit history in path update for each agent
            Solver_ILS.node_setup(**node_properties)

            # agents setup
            agent_properties = {'time_budget': t_max,
                                'origin': origin,
                                'destination': destination,
                                'preference': pref,
                                'visited': visit_history}

            Solver_ILS.agent_setup(**agent_properties)

            # each path_op will be saved into the predicted path set for agent n
            path_pdt = []

            # %% strat up solver
            no_init_flag = 0
            # solver initialization
            initial_path = Solver_ILS.initial_solution()

            # skip agents with empty initialized path
            if not initial_path:
                err_init.append(_idd)
                # initial_skip += 1
                # continue

            if len(initial_path) <= 2:
                no_init_flag = 1
            else:
                first_visit = initial_path[1]
                Solver_ILS.Node_list[first_visit].visit = 1

                order = initial_path
                final_order = list(order)

                # No edgeMethod in my case
                _u, _u8, _U10 = [], [], []

                counter_2 = 0
                no_improve = 0
                best_found = float('-inf')

                while no_improve < 50:
                    best_score = float('-inf')
                    local_optimum = 0

                    # print(Order)

                    while local_optimum == 0:
                        local_optimum, order, best_score = Solver_ILS.insert(order, best_score)

                    counter_2 += 1  # 2指inner loop的counter
                    v = len(order) - 1

                    _u.append(best_score)  # TODO U is utility memo
                    _u8.append(v)
                    _U10.append(max(_u))

                    if best_score > best_found:
                        best_found = best_score
                        final_order = list(order)

                        # save intermediate good paths into results
                        path_pdt.append(list(final_order))
                        no_improve = 0  # improved
                    else:
                        no_improve += 1

                    if len(order) <= 2:
                        continue
                    else:
                        s = np.random.randint(1, len(order) - 1)
                        R = np.random.randint(1, len(order) - s)  # from node S, delete R nodes (including S)

                    if s >= min(_u8):
                        s = s - min(_u8) + 1

                    order = Solver_ILS.shake(order, s, R)

            # print('Near optimal path: {}, with total time {} min, utility {}.'.format(final_order,
            #                                                                           Solver_ILS.time_callback(
            #                                                                               final_order),
            #                                                                           Solver_ILS.eval_util(
            #                                                                               final_order)))

            # Prediction penalty evaluation. Compare the predicted paths with observed one.

            path_obs = list(
                np.array(_agent.path_obs) - 1)  # attraction indices in solver start from 0 (in survey start from 1)

            # last modified on Oct. 24 16:29 2019
            # last modified on Dec. 20

            if no_init_flag:
                # do compulsory fill
                path_pdt.append(Solver_ILS.comp_fill())
                pass

            """对比的是combinatorial path score"""
            selected_path = []
            if len(path_pdt) < 3:  # return directly if...
                selected_path = list(path_pdt)
            else:
                # evaluate scores for all path predicted (not penalty with the observed path here)
                path_pdt_score = []
                for _path in path_pdt:
                    path_pdt_score.append(Solver_ILS.eval_util(_path))  # a list of penalties

                filter_ratio = 0.15  # predicted paths with penalties within 15% interval
                max_score = max(path_pdt_score)  # max utility score for current path

                ''' within 90% score or at least 3 paths in the predicted path set'''
                threshold = max_score - abs(filter_ratio * max_score)

                for _ in np.argsort(path_pdt_score)[::-1]:
                    if path_pdt_score[_] >= threshold:
                        selected_path.append(path_pdt[_])
                    else:
                        break
                # at least 3 paths in the set
                if len(selected_path) < 3:
                    selected_path = []
                    for _ in np.argsort(path_pdt_score)[-3:]:
                        selected_path.append(path_pdt[_])

            # -------- Path penalty evaluation --------
            # compare predicted path with observed ones
            # modified on Nov. 3 2019
            # last modified on Nov. 16

            best_path_predicted, lowest_penalty = [], float('inf')
            for _path in selected_path:
                res = Solver_ILS.path_penalty(path_obs, _path)
                if not best_path_predicted:
                    best_path_predicted, lowest_penalty = _path, res
                if res < lowest_penalty:
                    best_path_predicted, lowest_penalty = _path, res
                # print('With path score: %.2f, time: %d' % (solver.eval_util(_path, Pref).
                # solver.time_callback(_path)))
                # print('Penalty: {}'.format(res))
                # print_path(_path)

            # WRITE PREDICTED PATH AND PENALTY

            _penalty.append(lowest_penalty)
            _pdt_path.append(best_path_predicted)
            _obs_path.append(path_obs)

            end_time = datetime.datetime.now()
            success_usr_cnt += 1
            success_set.add(_idd)
            t_passed = (end_time - start_time).seconds
            if t_passed > 60:
                print('------ Evaluation time: {}s for agent id {}------\n'.format(t_passed, _idd))
            # update progress bar

        # todo 加上现在的lowest_penalty，可以先对penalty进行argsort然后取index
        # TODO create a dict? tuple (idx, data) . data includes penalty,
        #  [top 10 pdt_paths, least 10 pdt_paths], [top 10 obs_paths, least 10 obs_paths]

        sorted_indices = np.argsort(_penalty)  # according to mismatch penalty, for all tourists, from min to max
        predicted = [_pdt_path[_] for _ in sorted_indices[:10]] + [_pdt_path[_] for _ in sorted_indices[-10:]]
        observed = [_obs_path[_] for _ in sorted_indices[:10]] + [_obs_path[_] for _ in sorted_indices[-10:]]

        data = {'penalty': sum(_penalty) / 1000, 'predicted': predicted,
                'observed': observed}  # unit of penalty transformed from 'meter' to 'kilometer'.

        print('For process {}, successfully evaluated {} users. {} users were skipped.'.
              format(process_idx, success_usr_cnt, len(agent_database) - success_usr_cnt, ))

        q.put((process_idx, data['penalty']))

    @staticmethod
    def solver_debug_backup(process_idx, node_num, agent_database, **kwargs):  #
        # pass variables
        alpha, beta, phi, util_matrix, time_matrix, cost_matrix, dwell_matrix, dist_matrix = kwargs['alpha'], \
                                                                                             kwargs['beta'], kwargs[
                                                                                                 'phi'], kwargs[
                                                                                                 'util_matrix'], kwargs[
                                                                                                 'time_matrix'], kwargs[
                                                                                                 'cost_matrix'], kwargs[
                                                                                                 'dwell_matrix'], \
                                                                                             kwargs[
                                                                                                 'dist_matrix']

        # behavioral parameters data setup

        Solver_ILS.alpha = kwargs['alpha']
        Solver_ILS.beta = kwargs['beta']
        Solver_ILS.phi = kwargs['phi']

        # save results for all agents
        _penalty, _pdt_path, _obs_path = [], [], []

        # enumerate each tourist
        # node setup

        node_properties = {'node_num': node_num,
                           'utility_matrix': util_matrix,
                           'dwell_vector': dwell_matrix}

        # edge setup

        edge_properties = {'edge_time_matrix': time_matrix,
                           'edge_cost_matrix': cost_matrix,
                           'edge_distance_matrix': dist_matrix}

        Solver_ILS.edge_setup(**edge_properties)

        iteration_size = len(agent_database)

        # enumerate all tourists
        success_usr_cnt = 0
        success_set = set()
        initial_skip = 0

        for _idd, _agent in enumerate(agent_database):
            if _idd > 0 and _idd % 500 == 0:
                print(
                    '--- Running optimal tours for the {} agent in {} for process {}'.format(
                        _idd, len(agent_database), mp.current_process().name))

            pref = _agent.preference
            observed_path = _agent.path_obs
            t_max, origin, destination = _agent.time_budget, observed_path[0] - 1, observed_path[-1] - 1

            # todo
            """ 如果要考虑multiple-day travel,则可以加一个for _ in range(_agent.day_of_travel)，
            每一天的predicted path加入visit_history"""
            visit_history = {}

            if pref is None or observed_path is None:
                continue
            # skip empty paths (no visited location)
            if len(observed_path) < 3:
                continue

            start_time = datetime.datetime.now()

            """node setup process should be here!!"""
            # ... since nodes controls visit history in path update for each agent
            Solver_ILS.node_setup(**node_properties)

            # agents setup
            agent_properties = {'time_budget': t_max,
                                'origin': origin,
                                'destination': destination,
                                'preference': pref,
                                'visited': visit_history}

            Solver_ILS.agent_setup(**agent_properties)

            # each path_op will be saved into the predicted path set for agent n
            path_pdt = []

            # %% strat up solver
            # solver initialization
            initial_path = Solver_ILS.initial_solution()

            # skip agents with empty initialized path
            if not initial_path:
                initial_skip += 1
                continue

            if len(initial_path) <= 2:
                final_order = initial_path
            else:
                first_visit = initial_path[1]
                Solver_ILS.Node_list[first_visit].visit = 1

                order = initial_path
                final_order = list(order)

                # No edgeMethod in my case
                _u, _u8, _U10 = [], [], []

                counter_2 = 0
                no_improve = 0
                best_found = float('-inf')

                while no_improve < 50:
                    best_score = float('-inf')
                    local_optimum = 0

                    # print(Order)

                    while local_optimum == 0:
                        local_optimum, order, best_score = Solver_ILS.insert(order, best_score)

                    counter_2 += 1  # 2指inner loop的counter
                    v = len(order) - 1

                    _u.append(best_score)  # TODO U is utility memo
                    _u8.append(v)
                    _U10.append(max(_u))

                    if best_score > best_found:
                        best_found = best_score
                        final_order = list(order)

                        # save intermediate good paths into results
                        path_pdt.append(list(final_order))
                        no_improve = 0  # improved
                    else:
                        no_improve += 1

                    if len(order) <= 2:
                        continue
                    else:
                        s = np.random.randint(1, len(order) - 1)
                        R = np.random.randint(1, len(order) - s)  # from node S, delete R nodes (including S)

                    if s >= min(_u8):
                        s = s - min(_u8) + 1

                    order = Solver_ILS.shake(order, s, R)

            # print('Near optimal path: {}, with total time {} min, utility {}.'.format(final_order,
            #                                                                           Solver_ILS.time_callback(
            #                                                                               final_order),
            #                                                                           Solver_ILS.eval_util(
            #                                                                               final_order)))

            # Prediction penalty evaluation. Compare the predicted paths with observed one.

            path_obs = list(
                np.array(_agent.path_obs) - 1)  # attraction indices in solver start from 0 (in survey start from 1)

            # last modified on Oct. 24 16:29 2019

            """对比的是combinatorial path score"""
            selected_path = []
            if len(path_pdt) < 3:  # return directly if...
                selected_path = list(path_pdt)
            else:
                # evaluate scores for all path predicted (not penalty with the observed path here)
                path_pdt_score = []
                for _path in path_pdt:
                    path_pdt_score.append(Solver_ILS.eval_util(_path))  # a list of penalties

                filter_ratio = 0.15  # predicted paths with penalties within 15% interval
                max_score = max(path_pdt_score)  # max utility score for current path

                ''' within 90% score or at least 3 paths in the predicted path set'''
                threshold = max_score - abs(filter_ratio * max_score)

                for _ in np.argsort(path_pdt_score)[::-1]:
                    if path_pdt_score[_] >= threshold:
                        selected_path.append(path_pdt[_])
                    else:
                        break
                # at least 3 paths in the set
                if len(selected_path) < 3:
                    selected_path = []
                    for _ in np.argsort(path_pdt_score)[-3:]:
                        selected_path.append(path_pdt[_])

            # -------- Path penalty evaluation --------
            # compare predicted path with observed ones
            # modified on Nov. 3 2019
            # last modified on Nov. 16

            best_path_predicted, lowest_penalty = [], float('inf')
            for _path in selected_path:
                res = Solver_ILS.path_penalty(path_obs, _path)
                if not best_path_predicted:
                    best_path_predicted, lowest_penalty = _path, res
                if res < lowest_penalty:
                    best_path_predicted, lowest_penalty = _path, res
                # print('With path score: %.2f, time: %d' % (solver.eval_util(_path, Pref).
                # solver.time_callback(_path)))
                # print('Penalty: {}'.format(res))
                # print_path(_path)

            # WRITE PREDICTED PATH AND PENALTY

            _penalty.append(lowest_penalty)
            _pdt_path.append(best_path_predicted)
            _obs_path.append(path_obs)

            end_time = datetime.datetime.now()
            success_usr_cnt += 1
            success_set.add(_idd)
            t_passed = (end_time - start_time).seconds
            if t_passed > 60:
                print('------ Evaluation time: {}s for agent id {}------\n'.format(t_passed, _idd))
            # update progress bar

        # todo 加上现在的lowest_penalty，可以先对penalty进行argsort然后取index
        # TODO create a dict? tuple (idx, data) . data includes penalty,
        #  [top 10 pdt_paths, least 10 pdt_paths], [top 10 obs_paths, least 10 obs_paths]

        sorted_indices = np.argsort(_penalty)  # according to mismatch penalty, for all tourists, from min to max
        predicted = [_pdt_path[_] for _ in sorted_indices[:10]] + [_pdt_path[_] for _ in sorted_indices[-10:]]
        observed = [_obs_path[_] for _ in sorted_indices[:10]] + [_obs_path[_] for _ in sorted_indices[-10:]]

        data = {'penalty': sum(_penalty) / 1000, 'predicted': predicted,
                'observed': observed}  # unit of penalty transformed from 'meter' to 'kilometer'.

        print('Successfully evaluated {} users. {} users were skipped, including {} of initial path.'.format(
            success_usr_cnt,
            len(agent_database) - success_usr_cnt, initial_skip))

        return process_idx, data, success_set  # for debug


if __name__ == '__main__':
    # debug
    # Data preparation
    # %% read place code
    place_jp = pd.read_excel(os.path.join(os.path.dirname(__file__), 'Database', 'Place_code.xlsx'))
    # %% read tourist agents
    with open(os.path.join(os.path.dirname(__file__), 'Database', 'transit_user_database.pickle'),
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
    print('Total {} error found\n'.format(cnt))
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
    # s_opt = [-1.286284872, -0.286449175, 0.691566901, 0.353739632]
    # Parameter 20: a1: -0.800, a2: -0.026; b2: 10.012, b3: 0.007, with score: 2.189e-01
    # 1/a2 = -38.46

    alpha = -50
    beta = {'intercept': 3000, 'shape': 7, 'scale': 0.5}

    """previous beta1 = 100, * 38.46 ~ 3000; exp_x = shape * scale = 3.5"""

    # pass variables
    util_matrix, time_matrix, cost_matrix, dwell_matrix, dist_matrix = \
        utility_matrix, edge_time_matrix, edge_cost_matrix, dwell_vector, edge_distance_matrix

    # behavioral parameters data setup

    Solver_ILS.alpha = alpha
    Solver_ILS.beta = beta
    Solver_ILS.phi = phi

    # save results for all agents
    _penalty, _pdt_path, _obs_path = [], [], []

    # enumerate each tourist
    # node setup

    node_properties = {'node_num': node_num,
                       'utility_matrix': util_matrix,
                       'dwell_vector': dwell_matrix}

    # edge setup

    edge_properties = {'edge_time_matrix': time_matrix,
                       'edge_cost_matrix': cost_matrix,
                       'edge_distance_matrix': dist_matrix}

    Solver_ILS.edge_setup(**edge_properties)

    pref = [0.5, 0.3, 0.3]  # just for test
    observed_path = [29, 27, 24, 25, 29]  # index starts from 1
    t_max, origin, destination = 340, observed_path[0] - 1, observed_path[-1] - 1

    visit_history = {}

    start_time = datetime.datetime.now()

    """node setup process should be here!!"""
    # ... since nodes controls visit history in path update for each agent
    Solver_ILS.node_setup(**node_properties)

    # agents setup
    agent_properties = {'time_budget': t_max,
                        'origin': origin,
                        'destination': destination,
                        'preference': pref,
                        'visited': visit_history}

    Solver_ILS.agent_setup(**agent_properties)

    # each path_op will be saved into the predicted path set for agent n
    path_pdt = []
    path_obs = list(np.array(observed_path) - 1)

    for _ in range(len(path_obs) - 1):
        edge_tmp = path_obs[_], path_obs[_ + 1]
        print('Edge travel utility of {}: {:.1f}'.format(edge_tmp,
                                                         Solver_ILS.arc_util_callback(edge_tmp[0], edge_tmp[1])))

    print('\n')
    Solver_ILS.eval_util_print(path_obs)

    # %% test path penalty function
    path_a, path_b = [28, 27, 26, 23, 24, 28], [28, 27, 23, 24, 28]
    path_d = [28, 20, 13, 22, 25, 28]
    path_c = [28, 22, 23, 24, 23, 28]

    # print paths
    print('\nPrint current paths:')
    tmp = list(path_a)
    print('Paht_a:')
    while tmp:
        cur = tmp.pop(0)
        print(place_jp.name[cur] + ' -> ', end=' ') if tmp else print(place_jp.name[cur])

    tmp = list(path_b)
    print('Paht_b:')
    while tmp:
        cur = tmp.pop(0)
        print(place_jp.name[cur] + ' -> ', end=' ') if tmp else print(place_jp.name[cur])

    penalty_LD = Solver_ILS.path_penalty(path_a, path_b)
    penalty_SimGeo = Solver_ILS.geo_dist_penalty(path_a, path_b)
    print('Penalty modified LD: {:.2f} m, Geo_dist: {} m'.format(penalty_LD, penalty_SimGeo))
