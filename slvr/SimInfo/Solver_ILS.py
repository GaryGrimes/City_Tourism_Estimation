"""
Comments: Modules of the TTDP solver. Each instance of this script represents for a path prediction procedure for a given
tourist under a set of behavioral parameters.
Functionality: Receive behavioral parameters and start up solver methods.
Last modified on Nov. 11. Last modified on Jan. 18.
 """

import numpy as np
import Agent
import Network
import math
from scipy.stats import gamma
from scipy.optimize import curve_fit
import warnings

# behavioral parameters
alpha = None  # single entry
beta = {'intercept': None, 'shape': None, 'scale': None, 'time': None}  # 1 * 3 vector
phi = None  # float, to transform monetary unit into time

# instances of nodes and egdes contains solver methods. Thus they are in the same category with solver methods.
Node_list = []
Edge_list = []

# boundaries and centers of attraction areas

area_centers = {1: [135.8246, 35.095],
                2: [135.764472, 35.1156085],
                3: [135.778122, 35.069874999999996],
                4: [135.75400000000002, 35.058],
                5: [135.675, 35.05715],
                6: [135.80105, 35.049549999999996],
                7: [135.73067450000002, 35.053399],
                8: [135.7637, 35.048950000000005],
                9: [135.74714999999998, 35.04375],
                10: [135.7295, 35.039550000000006],
                11: [135.77275, 35.03505],
                12: [135.73675, 35.03105],
                13: [135.7188, 35.031949999999995],
                14: [135.67745, 35.02485],
                15: [135.79863, 35.02693],
                16: [135.7942, 35.01745],
                17: [135.78395, 35.0169],
                18: [135.76330000000002, 35.025999999999996],
                19: [135.72289999999998, 35.022400000000005],
                20: [135.7489, 35.0143],
                21: [135.7394, 35.011449999999996],
                22: [135.70925, 35.01535],
                23: [135.67415, 35.01465],
                24: [135.77935000000002, 35.00375],
                25: [135.7656, 35.0051],
                26: [135.68455, 34.995999999999995],
                27: [135.7806, 34.99435],
                28: [135.77100000000002, 34.989149999999995],
                29: [135.7548, 34.9893],
                30: [135.71025, 34.98355],
                31: [135.7758, 34.97985],
                32: [135.7465, 34.981049999999996],
                33: [135.77625, 34.9675],
                34: [135.82065, 34.9511],
                35: [135.74743, 34.950225],
                36: [135.75799999999998, 34.9313],
                37: [135.66645, 35.16765],
                39: [135.7594209, 34.9963851],
                38: [135.7594209, 34.9963851],
                41: [135.7583234, 34.9853387],
                40: [135.7583234, 34.9853387],
                42: [135.73208, 34.98115],
                43: [135.76882, 35.00372],
                44: [135.77217, 35.00886],
                45: [135.761225, 34.945958],
                46: [[135.7923442, 35.0581761], [135.6814425, 35.0164451]],
                47: [135.7596385, 35.010865]}

fit_params = []

insertion_penalty = 100  # 100 times larger than substitution


def logit(x, a, b, c):  # Logistic B equation from zunzun.com
    return a / (1.0 + np.power(x / b, c))


def fit_logit(func):
    global fit_params
    # generate x,y pairs to fit dramma distribution cdf

    _intercept, _shape, _scale = beta['intercept'], beta['shape'], beta['scale']  # scale < 1
    xmin, xmax = 0, max(10, _shape * _scale * 2)  # double of the expected x
    x = np.linspace(xmin, xmax, 25 * int(abs(xmax - xmin)))

    y = 1 - gamma.cdf(x, a=_shape, scale=_scale)
    # logistic fit
    initialParameters = np.array([1.0, 1.0, 1.0])
    # curve fit the test data, ignoring warning due to initial parameter estimates
    warnings.filterwarnings("ignore")
    try:
        fit_params, pcov = curve_fit(func, x, y, initialParameters)
    except RuntimeError:
        raise RuntimeError('Runtime error at shape {} and scale {}'.format(_shape, _scale))


# -------- node setup -------- #
def node_setup(**kwargs):
    global Node_list

    Network.node_num = kwargs['node_num']  # Number of attractions, excluding Origin and destination.
    Network.util_mat = kwargs['utility_matrix']
    Network.dwell_vec = kwargs['dwell_vector']

    # for each agent do a node setup.
    # Thus the global variable Node_list and Edge list should be initialized for each agent.
    Node_list = []
    for count in range(Network.node_num):
        x = Network.Node(Network.dwell_vec[count], Network.util_mat[count], count)
        x.visit = 1 if count in Agent.visited else 0
        Node_list.append(x)


def edge_setup(**kwargs):
    global Edge_list
    Network.time_mat = kwargs['edge_time_matrix']  # Number of attractions, excluding Origin and destination.
    Network.cost_mat = kwargs['edge_cost_matrix']
    Network.dist_mat = kwargs['edge_distance_matrix']  # distance between attraction areas

    Edge_list = []
    for origin in range(Network.time_mat.shape[0]):
        edge_list = []
        for destination in range(Network.time_mat.shape[1]):
            x = Network.Edge(phi, origin, destination, Network.time_mat[origin, destination],
                             Network.cost_mat[origin, destination], Network.dist_mat[origin, destination])
            edge_list.append(x)
        Edge_list.append(edge_list)


def agent_setup(**kwargs):
    Agent.t_max = kwargs['time_budget']  # person specific time constraints
    Agent.Origin = kwargs['origin']
    Agent.Destination = kwargs['destination']
    Agent.pref = kwargs['preference']
    Agent.visited = kwargs['visited']


# -------- model formulation (utility) setup -------- #

def arc_util_callback(from_node, to_node):
    # u(travel) = - t_ij - alpha_1 * c_ij. alpha_1 should be positive.
    # arc utility must at least be zero
    return -Network.time_mat[from_node, to_node] - max(alpha * Network.cost_mat[from_node, to_node], 0)


def node_util_callback(to_node, _accum_util):
    # modified on Jan. 17
    global beta
    _intercept, _shape, _scale = beta['intercept'], beta['shape'], beta['scale']  # scale < 1
    # always presume a negative discount factor
    _visit_util = exp_util_callback(to_node, _accum_util)

    if 'time' in beta:
        return _intercept * np.dot(Agent.pref, _visit_util) + beta['time'] * Network.dwell_vec[to_node]
    return _intercept * np.dot(Agent.pref, _visit_util)


def exp_util_callback(visit_node, cumu_util):
    """Now using fitted logistic curve to approximate the cdf of gamma distribution."""
    return Network.util_mat[visit_node] * logit(cumu_util, *fit_params)


def eval_util(_route):  # use array as input. The memoizer must accept agent's preference as well.
    res, _accum_util = 0, np.zeros([3])
    if len(_route) <= 2:
        return float("-inf")
    else:
        _k = 1
        for _k in range(1, len(_route) - 1):
            # arc and node utility
            res += arc_util_callback(_route[_k - 1], _route[_k]) + node_util_callback(_route[_k], _accum_util)
            _accum_util += exp_util_callback(_route[_k], _accum_util)  # Accumulated utility; travel history
        res += arc_util_callback(_route[_k], _route[_k + 1])
        return res
    pass


def eval_util_print(_route):  # use array as input
    """Print detailed route utility gains by segments"""
    print('Current route is {}. In evaluation. below indices start from 0'.format(list(np.array(_route) + 1)))
    _pref = Agent.pref
    res, _accum_util = 0, np.zeros([3])

    print('Printing detailed route utility gains...\n')
    if len(_route) <= 2:
        return float("-inf")
    else:
        _k = 1
        for _k in range(1, len(_route) - 1):
            # arc and node utility
            edge_cost = arc_util_callback(_route[_k - 1], _route[_k])
            node_visit_gain = node_util_callback(_route[_k], _accum_util)
            print('{}: Currrent travel {} costs {:.2f}, node visit {} gains {:.2f}'.format(_k,
                                                                                           (_route[_k - 1], _route[_k]),
                                                                                           edge_cost, _route[_k],
                                                                                           node_visit_gain))
            res += edge_cost + node_visit_gain
            print('{}: Current utility gained: {:.2f}'.format(_k, res))

            _accum_util += exp_util_callback(_route[_k], _accum_util)  # Accumulated utility; travel history
            print('{}: Current accumulated utility: {}\n'.format(_k, _accum_util))

        edge_cost = arc_util_callback(_route[_k], _route[_k + 1])
        res += edge_cost
        print('{}: Final travel {} costs {:.2f}'.format(_k + 1, (_route[_k], _route[_k + 1]), edge_cost))
        print('Final score: {:.2f}'.format(res))
    pass


def travel_time_callback(from_node, to_node):
    return Network.time_mat[from_node, to_node]


def time_callback(_route):
    _DwellArray = Network.dwell_vec
    if len(_route) <= 2:
        return 0
    else:
        _time, _k = 0, 1
        for _k in range(1, len(_route) - 1):
            _time += travel_time_callback(_route[_k - 1], _route[_k]) + _DwellArray[_route[_k]]
        _time += travel_time_callback(_route[_k], _route[_k + 1])
        return _time


def cost_change(self, n1, n2, n3, n4):
    cost_matrix = self.costmatrix
    return cost_matrix[n1][n3] + cost_matrix[n2][n4] - cost_matrix[n1][n2] - cost_matrix[n3][n4]


def util_change(n1, n2, n3, n4):  # utility improvement if result is positive
    return arc_util_callback(n1, n3) + arc_util_callback(n2, n4) - arc_util_callback(n1, n2) - arc_util_callback(n3, n4)


def initial_solution():
    o, d, t_max = Agent.Origin, Agent.Destination, Agent.t_max

    distance, benefit = [], []
    for _i in range(Network.node_num):
        # cost = Network.time_mat[o, _i] + Network.time_mat[_i, d]  # + Network.dwell_vec[_i]
        """updated on Jan. 21"""
        if o == d == _i:
            cost = 0.00001
        else:
            cost = abs(arc_util_callback(o, _i) + arc_util_callback(_i, d))  # cost have positive value
        distance.append(cost)

        _benefit = np.dot(Agent.pref, Network.util_mat[_i]) / cost
        benefit.append(_benefit)

    # index is sorted such that the first entry has smallest benefit for insertion (from o to d)
    index = list(np.argsort(benefit))
    # except for node_j

    # check time limitation
    # nodes with higher benefits at front
    available_nodes = [_x for _x in index if time_callback([o, _x, d]) <= t_max][::-1]

    # if over time limit
    if not available_nodes:
        return []

    # Otherwise, randomly pick an available node for insertion
    initial_path, success = [], 1
    comp_travel_util = arc_util_callback(o, d)  # a compulsive cost for traveling from o to d
    while not initial_path or eval_util(initial_path) < comp_travel_util:  # no better choice than going directly to d
        if available_nodes:
            node_to_insert = available_nodes.pop(np.random.randint(len(available_nodes)))
            initial_path = [o, node_to_insert, d]
        else:
            success = 0
            break

    return initial_path if success else [o, d]


def comp_fill():
    o, d, t_max = Agent.Origin, Agent.Destination, Agent.t_max
    distance, benefit = [], []
    for _i in range(Network.node_num):
        # cost = Network.time_mat[o, _i] + Network.time_mat[_i, d] + Network.dwell_vec[_i]
        cost = abs(arc_util_callback(o, _i) + arc_util_callback(_i, d))
        distance.append(cost)

        _benefit = np.dot(Agent.pref, Network.util_mat[_i]) / cost
        benefit.append(_benefit)
    # index is sorted such that the first entry has smallest benefit for insertion (from o to d)
    index = list(np.argsort(benefit))

    # compulsory fill-in
    return [o, index[-1], d]


def insert(order, best_score):
    t_max = Agent.t_max
    local_optimum = 0
    best_node, best_pos = None, None
    check = best_score  # score record

    _feasibility = 1
    for ii in range(len(Node_list)):
        cur_node = Node_list[ii]
        if cur_node.visit == 0:
            for jj in range(1, len(order)):
                path_temp = order[:jj] + [ii] + order[jj:]  # node index is ii
                # check time budget feasibility
                _feasibility = time_callback(path_temp) <= t_max
                # calculate utility and save best score and best position
                if _feasibility:
                    _utility = eval_util(path_temp)
                    if _utility > best_score:  # update
                        best_score, best_node, best_pos = _utility, ii, jj

    if best_score > check:
        order = order[:best_pos] + [best_node] + order[best_pos:]
        # update the node list: flag the visited nodes
        for ii in range(1, len(order) - 1):
            Node_list[order[ii]].visit = 1
    else:
        local_optimum = 1
    return local_optimum, order, best_score


def shake(order, s, r):
    path_temp = order[:s] + order[s + r:]
    # delete node visits
    for _ in range(s, s + r):
        Node_list[order[_]].visit = 0
    return path_temp


def haver_dist(lon1, lat1, lon2, lat2):
    """Input: geological coordinates of two locations, in a list [lon, lat]. Output:
    Eculidian distance between the two locations in meters."""

    b = math.pi / 180
    c = math.sin((lat2 - lat1) * b / 2)
    d = math.sin((lon2 - lon1) * b / 2)
    a = c * c + d * d * math.cos(lat1 * b) * math.cos(lat2 * b)
    return 12756274 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def find_path_center(_path):
    """find geological center of all visited nodes in the observed path (center of gravity)"""

    _saved, _to_decide = [], []
    for _ in _path:
        _visit_norm = _ + 1  # destinations from the survey start from 1
        if isinstance(area_centers[_visit_norm][0], float):
            _saved.append(area_centers[_visit_norm])
        else:
            _to_decide.extend(area_centers[_visit_norm])
    # find most possible transit entrance by taking the one closer to the path center
    _cur_center = [np.mean(np.array(_saved)[:, 0]), np.mean(np.array(_saved)[:, 1])]  # long, lat

    _res, _dis = None, None
    for _node in _to_decide:
        temp = haver_dist(*_cur_center, *_node)
        if not _dis:
            _dis = temp
            _res = _node
        else:
            if temp < _dis:
                _dis = temp
                _res = _node
    # combine
    if _res:
        _saved.append(_res)

    # calculate center
    _final_center = [np.mean(np.array(_saved)[:, 0]), np.mean(np.array(_saved)[:, 1])]  # lon, lat
    return _final_center


""" Methods for evaluating penalty for the null case. Where the TTDP degrades to a typical OP problem."""


def initial_solution_null():
    o, d, t_max = Agent.Origin, Agent.Destination, Agent.t_max

    distance, benefit = [], []
    for _i in range(Network.node_num):
        # cost = Network.time_mat[o, _i] + Network.time_mat[_i, d]  # + Network.dwell_vec[_i]
        """updated on Jan. 21"""
        if o == d == _i:
            cost = 0.00001
        else:
            cost = abs(arc_util_callback(o, _i) + arc_util_callback(_i, d))  # cost have positive value
        distance.append(cost)

        _benefit = np.dot(Agent.pref, Network.util_mat[_i]) / cost  # Agent.pref will degrade to [1,1,1]
        benefit.append(_benefit)

    # index is sorted such that the first entry has smallest benefit for insertion (from o to d)
    index = list(np.argsort(benefit))
    # except for node_j

    # check time limitation
    # nodes with higher benefits at front
    available_nodes = [_x for _x in index if time_callback([o, _x, d]) <= t_max][::-1]

    # if over time limit
    if not available_nodes:
        return []

    # Otherwise, randomly pick an available node for insertion
    initial_path, success = [], 1
    comp_travel_util = arc_util_callback(o, d)  # a compulsive cost for traveling from o to d
    while not initial_path or eval_util_null(
            initial_path) < comp_travel_util:  # no better choice than going directly to d
        if available_nodes:
            node_to_insert = available_nodes.pop(np.random.randint(len(available_nodes)))
            initial_path = [o, node_to_insert, d]
        else:
            success = 0
            break

    return initial_path if success else [o, d]


def node_util_callback_null(to_node):
    # modified on Jan. 17
    global beta
    _intercept, _shape, _scale = beta['intercept'], beta['shape'], beta['scale']  # scale < 1
    # always presume a negative discount factor
    _visit_util = exp_util_callback_null(to_node)

    if 'time' in beta:
        return _intercept * np.dot(Agent.pref, _visit_util) + beta['time'] * Network.dwell_vec[to_node]
    return _intercept * np.dot(Agent.pref, _visit_util)


def exp_util_callback_null(visit_node):
    # no diminishing marginal utility, no "fatigue"
    return Network.util_mat[visit_node]


def eval_util_null(_route):  # use array as input. The memoizer must accept agent's preference as well.
    res = 0
    if len(_route) <= 2:
        return float("-inf")
    else:
        _k = 1
        for _k in range(1, len(_route) - 1):
            # arc and node utility
            res += arc_util_callback(_route[_k - 1], _route[_k]) + node_util_callback_null(_route[_k])
        res += arc_util_callback(_route[_k], _route[_k + 1])
        return res
    pass


def insert_null(order, best_score):
    t_max = Agent.t_max
    local_optimum = 0
    best_node, best_pos = None, None
    check = best_score  # score record

    _feasibility = 1
    for ii in range(len(Node_list)):
        cur_node = Node_list[ii]
        if cur_node.visit == 0:
            for jj in range(1, len(order)):
                path_temp = order[:jj] + [ii] + order[jj:]  # node index is ii
                # check time budget feasibility
                _feasibility = time_callback(path_temp) <= t_max
                # calculate utility and save best score and best position
                if _feasibility:
                    _utility = eval_util_null(path_temp)
                    if _utility > best_score:  # update
                        best_score, best_node, best_pos = _utility, ii, jj

    if best_score > check:
        order = order[:best_pos] + [best_node] + order[best_pos:]
        # update the node list: flag the visited nodes
        for ii in range(1, len(order) - 1):
            Node_list[order[ii]].visit = 1
    else:
        local_optimum = 1
    return local_optimum, order, best_score


""" below  defines and adjusts prediction error criteria #"""


def path_penalty(path_obs, path_pdt):
    """Calculates the difference between observed and predicted path in meters.
    Insertion cost is defined as the haversine distance between inserted nodes and the center of observed path."""

    distance_matrix = Network.dist_mat
    # node indices in path_a and path_b are both standardised, i.e. starting from 0.
    if path_obs[0] != path_pdt[0] or path_obs[-1] != path_pdt[-1]:
        raise ValueError('Paths have different o or d.')

    # define insertion cost
    o, d = path_obs[0], path_obs[-1]

    # ----- construction ---- #

    # insertion_cost = [distance_matrix[o][_] + distance_matrix[_][d] for _ in range(distance_matrix.shape[0])]

    # compare the distance between the center of observed path and the inserted node as the insertion cost
    center_obs = find_path_center(path_obs)

    insertion_cost = insertion_penalty * np.array(
        [haver_dist(*center_obs, *area_centers[_ + 1]) for _ in range(len(Node_list))])

    # check empty path
    path_a, path_b = path_obs[1:-1], path_pdt[1:-1]
    if not path_a or not path_b:
        if not path_a and not path_b:  # if all empty
            return 0
        elif path_a:  # path b is empty
            try:
                _ = sum([max(distance_matrix[x]) for x in path_a])  # 19-10-03: take the largest distance
            except IndexError:
                _ = 0
            return _
        else:  # path a is empty
            try:
                _ = sum([max(distance_matrix[x]) for x in path_b])  # calculate most distant results
            except IndexError:
                _ = 0
            return _

    # check node indices. Observed path with Detailed location (58) or unclear places (99) were skipped.

    rows, cols = len(path_a) + 1, len(path_b) + 1

    # the editing distance matrix
    dist = np.array([[0 for _ in range(cols)] for _ in range(rows)])
    # source prefixes can be transformed into empty strings

    # by deletions:
    for row in range(1, rows):
        dist[row][0] = dist[row - 1][0] + insertion_cost[path_a[row - 1]]
    # target prefixes can be created from an empty source string

    # by inserting the characters
    for col in range(1, cols):
        dist[0][col] = dist[0][col - 1] + insertion_cost[path_b[col - 1]]

    for col in range(1, cols):
        for row in range(1, rows):
            deletes = insertion_cost[path_a[row - 1]]
            inserts = insertion_cost[path_b[col - 1]]
            subs = distance_matrix[path_a[row - 1]][path_b[col - 1]]  # dist from a to b

            dist[row][col] = min(dist[row - 1][col] + deletes,
                                 dist[row][col - 1] + inserts,
                                 dist[row - 1][col - 1] + subs)  # substitution
    # balance the result by dividing (insertion_penalty + 1)
    return dist[row][col] / (insertion_penalty + 1)

    # TODO case when path length equal


def geo_dist_penalty(p_a, p_b):  # created on Nov.3 2019
    """Pa is observed path, Pb predicted"""

    # Offset is 0 for the 1st destination.
    distance_matrix = Network.dist_mat
    if p_a[0] != p_b[0] or p_a[-1] != p_b[-1]:
        raise ValueError('Paths have different o or d.')

    # define the penalty in utility form for every two destinations. u_ik stands for the generalized cost of travel
    o, d = p_a[0], p_a[-1]

    path_a, path_b = p_a[1:-1], p_b[1:-1]  # excluding origin and destination

    path_node_check = []
    for _path in [path_a, path_b]:
        _new_path = []
        for node in _path:
            if node <= min(distance_matrix.shape) - 1:
                _new_path.append(node)
        path_node_check.append(_new_path)
    path_a, path_b = path_node_check[0], path_node_check[1]

    # utility (negative) penalty evaluation
    cost, a, b = 0, o, o  # let a, b be origin

    # if exist empty path
    if not path_a:  # if observed path is empty
        return cost

    while path_a and path_b:
        a, b = path_a.pop(0), path_b.pop(0)  # a, b correspond to the i_th node in path_a, path_b
        cost += distance_matrix[a][b]

    if path_a:  # length of path_a > path b
        while path_a:
            a = path_a.pop(0)
            cost += distance_matrix[a][b]
    else:  # case when length of path_b > path a
        while path_b:
            b = path_b.pop(0)
            cost += distance_matrix[a][b]
    return cost


# def path_penalty(p_a, p_b):
#     distance_matrix = Network.dist_mat
#     # path_a and path_b both starts from 0. offset starts from 0, i.e., 0 --> 1st destination
#     if p_a[0] != p_b[0] or p_a[-1] != p_b[-1]:
#         raise ValueError('Paths have different o or d.')
#
#     # define insertion cost
#     o, d = p_a[0], p_a[-1]
#     insertion_cost = [distance_matrix[o][_] + distance_matrix[_][d] for _ in range(distance_matrix.shape[0])]
#
#     # check empty path
#     path_a, path_b = p_a[1:-1], p_b[1:-1]
#     if not path_a or not path_b:
#         if not path_a and not path_b:  # if all empty
#             return 0
#         elif path_a:  # path b is empty
#             try:
#                 _ = sum([max(distance_matrix[x]) for x in path_a])  # 19-10-03: take the largest distance
#             except IndexError:
#                 _ = 0
#             return _
#         else:  # path a is empty
#             try:
#                 _ = sum([max(distance_matrix[x]) for x in path_b])  # calculate most distant results
#             except IndexError:
#                 _ = 0
#             return _
#
#     # if both paths are not empty (excluding o, d)
#
#     # check node indices. Observed path with Detailed location (58) or unclear places (99) were skipped.
#     # TODO Better to omit outbound visits in the path rather than skip to next tourist. Completed in DataWrapping.
#     # max_idx = max(max(path_a), max(path_b))
#     # if max_idx > min(distance_matrix.shape) - 1:
#     #     return 0
#
#     rows, cols = len(path_a) + 1, len(path_b) + 1
#
#     # the editing distance matrix
#     dist = [[0 for _ in range(cols)] for _ in range(rows)]
#     # source prefixes can be transformed into empty strings
#
#     # by deletions:
#     for row in range(1, rows):
#         dist[row][0] = dist[row - 1][0] + insertion_cost[path_a[row - 1]]
#     # target prefixes can be created from an empty source string
#
#     # by inserting the characters
#     for col in range(1, cols):
#         dist[0][col] = dist[0][col - 1] + insertion_cost[path_b[col - 1]]
#
#     for col in range(1, cols):
#         for row in range(1, rows):
#             deletes = insertion_cost[path_a[row - 1]]
#             inserts = insertion_cost[path_b[col - 1]]
#             subs = distance_matrix[path_a[row - 1]][path_b[col - 1]]  # dist from a to b
#
#             dist[row][col] = min(dist[row - 1][col] + deletes,
#                                  dist[row][col - 1] + inserts,
#                                  dist[row - 1][col - 1] + subs)  # substitution
#     return dist[row][col]
#
#     # TODO case when path length equal


if __name__ == '__main__':
    # %% create node instances
    # assign values to node instances
    print(find_path_center([45, 23, 27, 29, 21, 40]))
    pass

    # test logit fit function
    range_alpha = [-3]
    range_intercept = [10]

    range_shape = [0.1, 0.5, 1, 2, 3, 5, 7, 9]
    range_scale = [0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 5, 10]
    inn = 0
    for i in range_shape:
        for j in range_scale:
            inn += 1
            beta = {'intercept': 10, 'shape': i, 'scale': j, 'time': None}  # 1 * 3 vector
            try:
                fit_logit(logit)
            except:
                print('Run time error at shape{}, scale{}, index {}'.format(i, j, inn % 16))
