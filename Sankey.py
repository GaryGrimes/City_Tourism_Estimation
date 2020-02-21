"""Test functionality of Sankey Diagram Plot.
Reference: https://zhuanlan.zhihu.com/p/80410924

Data should be in a structure like: subcategory-main category-flow

"""
import pandas as pd
import numpy as np
from pyecharts.charts import Page, Sankey
from pyecharts import options as opts
import plotly.graph_objects as go
from pyecharts.render import make_snapshot


def sankey_echart(trip_chain_df, target_node_idx, place_labels_jp, n_largest=7):
    """ illustration for a single node."""
    if n_largest < 1:
        raise ValueError('Please define valid threshold!')

    """ To plot independent previous and next patterns."""
    node_num = 37  # totally 37 areas in the survey
    if len(str(target_node_idx)) == 0:
        raise ValueError('Please input valid node index!')
    if target_node_idx > node_num - 1:
        raise IndexError('Node index exceeds the total number of attraction areas!')

    nodes_involved = set()

    # generate nodes data structure for Sankey plot input
    _nodes = []
    edges_from, edges_to = {}, {}
    for _trip in trip_chain_df:
        if target_node_idx in _trip:
            for _pointer in range(len(_trip) - 1):
                _from, _to = _trip[_pointer], _trip[_pointer + 1]
                if _from == target_node_idx or _to == target_node_idx:
                    if _from == target_node_idx:
                        _from_label = place_labels_jp[_from]
                        _to_label = place_labels_jp[_to] + '_着'
                        _ = edges_from
                    else:
                        _from_label = place_labels_jp[_from] + '_発'
                        _to_label = place_labels_jp[_to]
                        _ = edges_to

                    nodes_involved.add(_from_label)
                    nodes_involved.add(_to_label)

                    edge_pattern = str(_from_label) + '-' + str(_to_label)
                    if edge_pattern in _:
                        _[edge_pattern]['value'] += 1
                    else:
                        _[edge_pattern] = {'source': _from_label, 'target': _to_label, 'value': 1}

    # generate the from and to dicts, sort the flows and merge the rest into "Others"
    """Reference"""
    # x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
    # sorted_x = {k: v for k, v in sorted(x.items(), key=lambda item: item[1])}

    sorted_edges_from = dict(sorted(edges_from.items(), key=lambda item: item[1]['value'], reverse=True)[:n_largest])
    # add "Others edge pattern
    if len(edges_from) > n_largest:
        sorted_edges_from[place_labels_jp[target_node_idx] + '-' + 'Others_着'] = {
            'source': place_labels_jp[target_node_idx],
            'target': 'Others_着',
            'value': sum(
                [_[1]['value'] for _ in
                 sorted(edges_from.items(), key=lambda item: item[1]['value'], reverse=True)[n_largest:]])
            # add the rest of the k, v tuples from sorted dict
        }

    sorted_edges_to = dict(sorted(edges_to.items(), key=lambda item: item[1]['value'], reverse=True)[:n_largest])
    # add "Others edge pattern
    if len(edges_to) > n_largest:
        sorted_edges_to['Others_発' + '-' + place_labels_jp[target_node_idx]] = {
            'source': 'Others_発',
            'target': place_labels_jp[target_node_idx],
            'value': sum(
                [_[1]['value'] for _ in
                 sorted(edges_from.items(), key=lambda item: item[1]['value'], reverse=True)[n_largest:]])
            # add the rest of the k, v tuples from sorted dict
        }

    # todo: a comparison between predicted and observed. Or between base and improved.
    # generate link data structure for Sankey plot input
    _links = []
    _nodes = set()

    # next is an independent functionality

    for edge_involve_dict in (sorted_edges_to, sorted_edges_from):
        for k, v in edge_involve_dict.items():
            # avoid within node trips
            if v['source'] != v['target']:
                _nodes.add(v['source']), _nodes.add(v['target'])
                _links.append(
                    {'source': v['source'],
                     'target': v['target'],
                     'value': v['value']}
                )

    nodes = [{"name": _} for _ in list(_nodes)]

    _res = (
        Sankey().add(
            'Number of trips',
            nodes,
            _links,
            linestyle_opt=opts.LineStyleOpts(opacity=0.2, curve=0.5, color="source", type_="dotted"),
            label_opts=opts.LabelOpts(position="right", ),
        ).set_global_opts(title_opts=opts.TitleOpts(title="A Sankey Diagram of Node {}".format(
            place_labels_jp[target_node_idx])))
    )
    # output visualization by html file
    _res.render('Sankey plot trial/result_{}.html'.format(place_labels_jp[target_node_idx]))


def sankey_echart_comparison(base_df, new_df, target_node_idx, place_labels_jp, n_largest=7):
    """ illustration for a single node."""
    name_base, name_new = ['_obs', '_pdt']
    if n_largest < 1:
        raise ValueError('Please define valid threshold!')

    """ To plot independent previous and next patterns."""
    node_num = 37  # totally 37 areas in the survey
    if len(str(target_node_idx)) == 0:
        raise ValueError('Please input valid node index!')
    if target_node_idx > node_num - 1:
        raise IndexError('Node index exceeds the total number of attraction areas!')

    # generate link data structure for Sankey plot input
    _links = []
    _nodes = set()

    # generate nodes data structure for Sankey plot input
    edges_from, edges_to = {}, {}
    for _trip in base_df:
        if target_node_idx in _trip:
            for _pointer in range(len(_trip) - 1):
                _from, _to = _trip[_pointer], _trip[_pointer + 1]
                if _from == target_node_idx or _to == target_node_idx:
                    if _from == target_node_idx:
                        _from_label = place_labels_jp[_from] + name_base
                        _to_label = place_labels_jp[_to] + '_着'
                        _ = edges_from
                    else:
                        _from_label = place_labels_jp[_from] + '_発'
                        _to_label = place_labels_jp[_to] + name_base
                        _ = edges_to

                    edge_pattern = str(_from_label) + '-' + str(_to_label)
                    if edge_pattern in _:
                        _[edge_pattern]['value'] += 1
                    else:
                        _[edge_pattern] = {'source': _from_label, 'target': _to_label, 'value': 1}

    # generate the from and to dicts, sort the flows and merge the rest into "Others"
    """Reference"""
    # x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
    # sorted_x = {k: v for k, v in sorted(x.items(), key=lambda item: item[1])}

    sorted_edges_from = dict(sorted(edges_from.items(), key=lambda item: item[1]['value'], reverse=True)[:n_largest])
    # add "Others edge pattern
    if len(edges_from) > n_largest:
        sorted_edges_from[place_labels_jp[target_node_idx] + name_base + '-' + 'Others_着'] = {
            'source': place_labels_jp[target_node_idx] + name_base,
            'target': 'Others_着',
            'value': sum(
                [_[1]['value'] for _ in
                 sorted(edges_from.items(), key=lambda item: item[1]['value'], reverse=True)[n_largest:]])
            # add the rest of the k, v tuples from sorted dict
        }

    sorted_edges_to = dict(sorted(edges_to.items(), key=lambda item: item[1]['value'], reverse=True)[:n_largest])
    # add "Others edge pattern
    if len(edges_to) > n_largest:
        sorted_edges_to['Others_発' + '-' + place_labels_jp[target_node_idx] + name_base] = {
            'source': 'Others_発',
            'target': place_labels_jp[target_node_idx] + name_base,
            'value': sum(
                [_[1]['value'] for _ in
                 sorted(edges_from.items(), key=lambda item: item[1]['value'], reverse=True)[n_largest:]])
            # add the rest of the k, v tuples from sorted dict
        }

    for edge_involve_dict in (sorted_edges_to, sorted_edges_from):
        for k, v in edge_involve_dict.items():
            # avoid within node trips
            if v['source'] != v['target']:
                _nodes.add(v['source']), _nodes.add(v['target'])
                _links.append(
                    {'source': v['source'],
                     'target': v['target'],
                     'value': v['value']}
                )

    # generate nodes data structure for Sankey plot input
    edges_from, edges_to = {}, {}
    for _trip in new_df:
        if target_node_idx in _trip:
            for _pointer in range(len(_trip) - 1):
                _from, _to = _trip[_pointer], _trip[_pointer + 1]
                if _from == target_node_idx or _to == target_node_idx:
                    if _from == target_node_idx:
                        _from_label = place_labels_jp[_from] + name_new
                        _to_label = place_labels_jp[_to] + '_着'
                        _ = edges_from
                    else:
                        _from_label = place_labels_jp[_from] + '_発'
                        _to_label = place_labels_jp[_to] + name_new
                        _ = edges_to

                    edge_pattern = str(_from_label) + '-' + str(_to_label)
                    if edge_pattern in _:
                        _[edge_pattern]['value'] += 1
                    else:
                        _[edge_pattern] = {'source': _from_label, 'target': _to_label, 'value': 1}

    # generate the from and to dicts, sort the flows and merge the rest into "Others"
    """Reference"""
    # x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
    # sorted_x = {k: v for k, v in sorted(x.items(), key=lambda item: item[1])}

    sorted_edges_from = dict(sorted(edges_from.items(), key=lambda item: item[1]['value'], reverse=True)[:n_largest])
    # add "Others edge pattern
    if len(edges_from) > n_largest:
        sorted_edges_from[place_labels_jp[target_node_idx] + name_new + '-' + 'Others_着'] = {
            'source': place_labels_jp[target_node_idx] + name_new,
            'target': 'Others_着',
            'value': sum(
                [_[1]['value'] for _ in
                 sorted(edges_from.items(), key=lambda item: item[1]['value'], reverse=True)[n_largest:]])
            # add the rest of the k, v tuples from sorted dict
        }

    sorted_edges_to = dict(sorted(edges_to.items(), key=lambda item: item[1]['value'], reverse=True)[:n_largest])
    # add "Others edge pattern
    if len(edges_to) > n_largest:
        sorted_edges_to['Others_発' + '-' + place_labels_jp[target_node_idx] + name_new] = {
            'source': 'Others_発',
            'target': place_labels_jp[target_node_idx] + name_new,
            'value': sum(
                [_[1]['value'] for _ in
                 sorted(edges_from.items(), key=lambda item: item[1]['value'], reverse=True)[n_largest:]])
            # add the rest of the k, v tuples from sorted dict
        }

    # todo: a comparison between predicted and observed. Or between base and improved.

    # next is an independent functionality

    for edge_involve_dict in (sorted_edges_to, sorted_edges_from):
        for k, v in edge_involve_dict.items():
            # avoid within node trips
            if v['source'] != v['target']:
                _nodes.add(v['source']), _nodes.add(v['target'])
                _links.append(
                    {'source': v['source'],
                     'target': v['target'],
                     'value': v['value']}
                )

    nodes = [{"name": _} for _ in list(_nodes)]

    _res = (
        Sankey().add(
            'Number of trips',
            nodes,
            _links,
            linestyle_opt=opts.LineStyleOpts(opacity=0.2, curve=0.5, color="source", type_="dotted"),
            label_opts=opts.LabelOpts(position="right", ),
        ).set_global_opts(title_opts=opts.TitleOpts(title="A Sankey Diagram of Node {}".format(
            place_labels_jp[target_node_idx])))
    )
    # output visualization by html file
    _res.render('Sankey plot compare/result_{}.html'.format(place_labels_jp[target_node_idx]))
    # make_snapshot(driver, _res.render(), "Sankey plot compare/A Sankey Diagram of Node {}.png".format(
    #     place_labels_jp[target_node_idx]))


def sankey_echart_strategy(base_df, new_df, target_node_idx, place_labels_jp, stg_name, n_largest=7):
    """ illustration for a single node."""
    name_base, name_new = ['-', '+(TDM)']
    if n_largest < 1:
        raise ValueError('Please define valid threshold!')

    """ To plot independent previous and next patterns."""
    node_num = 37  # totally 37 areas in the survey
    if len(str(target_node_idx)) == 0:
        raise ValueError('Please input valid node index!')
    if target_node_idx > node_num - 1:
        raise IndexError('Node index exceeds the total number of attraction areas!')

    # generate link data structure for Sankey plot input
    _links = []
    _nodes = set()

    # generate nodes data structure for Sankey plot input
    edges_from, edges_to = {}, {}
    for _trip in base_df:
        if target_node_idx in _trip:
            for _pointer in range(len(_trip) - 1):
                _from, _to = _trip[_pointer], _trip[_pointer + 1]
                if _from == target_node_idx or _to == target_node_idx:
                    if _from == target_node_idx:
                        _from_label = place_labels_jp[_from] + name_base
                        _to_label = place_labels_jp[_to] + '_着'
                        _ = edges_from
                    else:
                        _from_label = place_labels_jp[_from] + '_発'
                        _to_label = place_labels_jp[_to] + name_base
                        _ = edges_to

                    edge_pattern = str(_from_label) + '-' + str(_to_label)
                    if edge_pattern in _:
                        _[edge_pattern]['value'] += 1
                    else:
                        _[edge_pattern] = {'source': _from_label, 'target': _to_label, 'value': 1}

    # generate the from and to dicts, sort the flows and merge the rest into "Others"
    """Reference"""
    # x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
    # sorted_x = {k: v for k, v in sorted(x.items(), key=lambda item: item[1])}

    sorted_edges_from = dict(sorted(edges_from.items(), key=lambda item: item[1]['value'], reverse=True)[:n_largest])
    # add "Others edge pattern
    if len(edges_from) > n_largest:
        sorted_edges_from[place_labels_jp[target_node_idx] + name_base + '-' + 'Others_着'] = {
            'source': place_labels_jp[target_node_idx] + name_base,
            'target': 'Others_着',
            'value': sum(
                [_[1]['value'] for _ in
                 sorted(edges_from.items(), key=lambda item: item[1]['value'], reverse=True)[n_largest:]])
            # add the rest of the k, v tuples from sorted dict
        }

    sorted_edges_to = dict(sorted(edges_to.items(), key=lambda item: item[1]['value'], reverse=True)[:n_largest])
    # add "Others edge pattern
    if len(edges_to) > n_largest:
        sorted_edges_to['Others_発' + '-' + place_labels_jp[target_node_idx] + name_base] = {
            'source': 'Others_発',
            'target': place_labels_jp[target_node_idx] + name_base,
            'value': sum(
                [_[1]['value'] for _ in
                 sorted(edges_from.items(), key=lambda item: item[1]['value'], reverse=True)[n_largest:]])
            # add the rest of the k, v tuples from sorted dict
        }

    for edge_involve_dict in (sorted_edges_to, sorted_edges_from):
        for k, v in edge_involve_dict.items():
            # avoid within node trips
            if v['source'] != v['target']:
                _nodes.add(v['source']), _nodes.add(v['target'])
                _links.append(
                    {'source': v['source'],
                     'target': v['target'],
                     'value': v['value']}
                )

    # generate nodes data structure for Sankey plot input
    edges_from, edges_to = {}, {}
    for _trip in new_df:
        if target_node_idx in _trip:
            for _pointer in range(len(_trip) - 1):
                _from, _to = _trip[_pointer], _trip[_pointer + 1]
                if _from == target_node_idx or _to == target_node_idx:
                    if _from == target_node_idx:
                        _from_label = place_labels_jp[_from] + name_new
                        _to_label = place_labels_jp[_to] + '_着'
                        _ = edges_from
                    else:
                        _from_label = place_labels_jp[_from] + '_発'
                        _to_label = place_labels_jp[_to] + name_new
                        _ = edges_to

                    edge_pattern = str(_from_label) + '-' + str(_to_label)
                    if edge_pattern in _:
                        _[edge_pattern]['value'] += 1
                    else:
                        _[edge_pattern] = {'source': _from_label, 'target': _to_label, 'value': 1}

    # generate the from and to dicts, sort the flows and merge the rest into "Others"
    """Reference"""
    # x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
    # sorted_x = {k: v for k, v in sorted(x.items(), key=lambda item: item[1])}

    sorted_edges_from = dict(sorted(edges_from.items(), key=lambda item: item[1]['value'], reverse=True)[:n_largest])
    # add "Others edge pattern
    if len(edges_from) > n_largest:
        sorted_edges_from[place_labels_jp[target_node_idx] + name_new + '-' + 'Others_着'] = {
            'source': place_labels_jp[target_node_idx] + name_new,
            'target': 'Others_着',
            'value': sum(
                [_[1]['value'] for _ in
                 sorted(edges_from.items(), key=lambda item: item[1]['value'], reverse=True)[n_largest:]])
            # add the rest of the k, v tuples from sorted dict
        }

    sorted_edges_to = dict(sorted(edges_to.items(), key=lambda item: item[1]['value'], reverse=True)[:n_largest])
    # add "Others edge pattern
    if len(edges_to) > n_largest:
        sorted_edges_to['Others_発' + '-' + place_labels_jp[target_node_idx] + name_new] = {
            'source': 'Others_発',
            'target': place_labels_jp[target_node_idx] + name_new,
            'value': sum(
                [_[1]['value'] for _ in
                 sorted(edges_from.items(), key=lambda item: item[1]['value'], reverse=True)[n_largest:]])
            # add the rest of the k, v tuples from sorted dict
        }

    # todo: a comparison between predicted and observed. Or between base and improved.

    # next is an independent functionality

    for edge_involve_dict in (sorted_edges_to, sorted_edges_from):
        for k, v in edge_involve_dict.items():
            # avoid within node trips
            if v['source'] != v['target']:
                _nodes.add(v['source']), _nodes.add(v['target'])
                _links.append(
                    {'source': v['source'],
                     'target': v['target'],
                     'value': v['value']}
                )

    nodes = [{"name": _} for _ in list(_nodes)]

    _res = (
        Sankey().add(
            'Number of trips',
            nodes,
            _links,
            linestyle_opt=opts.LineStyleOpts(opacity=0.2, curve=0.5, color="source", type_="dotted"),
            label_opts=opts.LabelOpts(position="right", ),
        ).set_global_opts(title_opts=opts.TitleOpts(title="A Sankey Diagram of Node {}".format(
            place_labels_jp[target_node_idx])))
    )
    # output visualization by html file
    _res.render('Sankey plot compare/TDM_{}_at_{}.html'.format(stg_name, place_labels_jp[target_node_idx]))
    # make_snapshot(driver, _res.render(), "Sankey plot compare/A Sankey Diagram of Node {}.png".format(
    #     place_labels_jp[target_node_idx]))

    print('Compared strategy effect at node {}'.format(place_labels_jp[target_node_idx]))

def sankey_base(n=None, l=None) -> Sankey:
    if n:
        nodes = n
    else:
        nodes = [
            {"name": "category1"},
            {"name": "category2"},
            {"name": "category3"},
            {"name": "category4"},
            {"name": "category5"},
            {"name": "category6"},
        ]
    if l:
        links = l
    else:
        links = [
            {"source": "category1", "target": "category2", "value": 10},
            {"source": "category2", "target": "category3", "value": 15},
            {"source": "category3", "target": "category4", "value": 20},
            {"source": "category5", "target": "category6", "value": 25},
        ]
    c = (
        Sankey()
            .add(
            "sankey",
            nodes,
            links,
            linestyle_opt=opts.LineStyleOpts(opacity=0.2, curve=0.5, color="source"),
            label_opts=opts.LabelOpts(position="right"),
        )
            .set_global_opts(title_opts=opts.TitleOpts(title="Sankey-基本示例"))
    )

    return c


def sankey_complete_flow(trip_chain_df, target_node_idx, place_labels_jp):
    """ Illustrate complete trip chains that include the target node.
    Input: normalized node index, starting from 0, i.e node_index in survey -1."""
    node_num = 37  # totally 37 areas in the survey
    if len(str(target_node_idx)) == 0:
        raise ValueError('Please input valid node index!')
    if target_node_idx > node_num - 1:
        raise IndexError('Node index exceeds the total number of attraction areas!')

    nodes_involved = set()

    edge_involve_dict = {}

    for _trip in trip_chain_df:
        if target_node_idx in _trip[1:-1]:  # must exclude origin and dest.
            offset = _trip[1:].index(target_node_idx) + 1  # avoid attraction area appears as origin
            idxs = np.array(range(len(_trip))) - offset

            # enumerate all trip segments (o,d)
            for _pointer in range(len(_trip) - 1):
                _from, _to = _trip[_pointer], _trip[_pointer + 1]
                # only for the trips whose o and d are both involved
                if _pointer == offset - 1:
                    name_from, name_to = place_labels_jp[_from] + '_l' + str(idxs[_pointer]), place_labels_jp[_to]

                elif _pointer == offset:
                    name_from, name_to = place_labels_jp[_from], place_labels_jp[_to] + '_l' + str(idxs[_pointer + 1])
                else:
                    # normal case
                    name_from, name_to = place_labels_jp[_from] + '_l' + str(idxs[_pointer]), \
                                         place_labels_jp[_to] + '_l' + str(idxs[_pointer + 1])
                # add node
                nodes_involved.add(name_from)  # from
                nodes_involved.add(name_to)  # to

                # add edge
                edge_pattern = str(name_from) + '-' + str(name_to)
                if edge_pattern in edge_involve_dict:
                    edge_involve_dict[edge_pattern]['value'] += 1
                else:
                    edge_involve_dict[edge_pattern] = {'source': name_from, 'target': name_to, 'value': 1}
            # todo: consider add an 'end' node

        pass
    # enumerate all trips that
    # _nodes = [{'name': _} for _ in nodes_involved]

    _node_label = list(nodes_involved)

    # generate link data structure for Sankey plot input
    source, target, value = [], [], []
    for k, v in edge_involve_dict.items():
        # avoid within node trips
        # if v['source'] != v['target']:
        source.append(_node_label.index(v['source']))
        target.append(_node_label.index(v['target']))
        value.append(v['value'])

    fig = go.Figure(data=[go.Sankey(
        valueformat=".0f",
        node=dict(
            pad=15,
            thickness=15,
            line=dict(color="black", width=0.5),
            label=_node_label,
        ),
        link=dict(
            source=source,  # indices correspond to labels, eg A1, A2, A2, B1, ...
            target=target,
            value=value
        ))])

    fig.update_layout(title_text="Trip flow Sankey Diagram for attraction {}".format(
        place_labels_jp[target_node_idx]
    ), font_size=10)

    fig.show()

    # _res = (
    #     Sankey().add(
    #         'Number of trips',
    #         _nodes,
    #         _links,
    #         linestyle_opt=opts.LineStyleOpts(opacity=0.2, curve=0.5, color="source", type_="dotted"),
    #         label_opts=opts.LabelOpts(position="right", ),
    #     ).set_global_opts(title_opts=opts.TitleOpts(title="A Sankey Diagram of Node {}".format(target_node_idx + 1)))
    # )
    # # output visualization by html file
    # _res.render('Sankey plot trial 1/result.html')
    pass


def sankey_plotly(trip_chain_df, target_node_idx, place_labels_jp):
    """ Plot only the previous and next destination. That is what planners are concerned about.
    Input: normalized node index, starting from 0, i.e node_index in survey -1.
    """
    node_num = 37  # totally 37 areas in the survey
    if len(str(target_node_idx)) == 0:
        raise ValueError('Please input valid node index!')
    if target_node_idx > node_num - 1:
        raise IndexError('Node index exceeds the total number of attraction areas!')

    nodes_involved = set()
    edge_involve_dict = {}

    # first loop: append nodes that are included in trips consist of the target node
    for _trip in trip_chain_df:
        if target_node_idx in _trip:
            for _pointer in range(len(_trip) - 1):
                _from, _to = _trip[_pointer], _trip[_pointer + 1]
                if _from == target_node_idx or _to == target_node_idx:
                    if _from == target_node_idx:
                        _from_label = place_labels_jp[_from]
                        _to_label = place_labels_jp[_to] + '_着'
                    else:
                        _from_label = place_labels_jp[_from] + '_発'
                        _to_label = place_labels_jp[_to]

                    nodes_involved.add(_from_label)
                    nodes_involved.add(_to_label)

                    edge_pattern = str(_from_label) + '-' + str(_to_label)

                    if edge_pattern in edge_involve_dict:
                        edge_involve_dict[edge_pattern]['value'] += 1
                    else:
                        edge_involve_dict[edge_pattern] = {'source': _from_label, 'target': _to_label, 'value': 1}

    # generate link data structure for Sankey plot input
    _node_label = list(nodes_involved)

    source, target, value = [], [], []
    for k, v in edge_involve_dict.items():
        # avoid within node trips
        # if v['source'] != v['target']:
        source.append(_node_label.index(v['source']))
        target.append(_node_label.index(v['target']))
        value.append(v['value'])

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=15,
            line=dict(color="black", width=0.5),
            label=_node_label,
        ),
        link=dict(
            source=source,  # indices correspond to labels, eg A1, A2, A2, B1, ...
            target=target,
            value=value
        ))])

    fig.update_layout(title_text="Sankey Diagram for attraction {}".format(
        place_labels_jp[target_node_idx]
    ), font_size=10)

    fig.show()


if __name__ == '__main__':
    # 导入相关库

    nodes = [
        {'name': '大原・八瀬方面'},
        {'name': '京都市内の自宅・知人宅'},
        {'name': '京都市内の宿泊施設'},
        {'name': 'ＪＲ京都駅（新幹線）'},
        {'name': 'ＪＲ京都駅（在来線）'},
        {'name': 'その他ＪＲ駅'},
        {'name': '下鴨神社周辺'},
        {'name': '京阪駅'},
        {'name': '平安神宮周辺'},
        {'name': '祇園方面'},
        {'name': '河原町・新京極方面'}
    ]

    # cannot have o-d and d-o at the same time
    links = [
        {'source': 'ＪＲ京都駅（新幹線）', 'target': '祇園方面', 'value': 239},
        {'source': '祇園方面', 'target': '河原町・新京極方面', 'value': 795},
        {'source': '京都市内の自宅・知人宅', 'target': '祇園方面', 'value': 71},
        {'source': 'ＪＲ京都駅（在来線）', 'target': '祇園方面', 'value': 223},
    ]

    test = sankey_base(n=nodes, l=links)
    test.render('test sankey.html')  # 可以通过配置项otps来设置图表的颜色、标签、标题等信息，具体细节可以去官网查询，这里不做赘述
    pass

    """
        links = [
        {'source': 'ＪＲ京都駅（新幹線）', 'target': '祇園方面', 'value': 239},
        {'source': '祇園方面', 'target': '河原町・新京極方面', 'value': 795},
        {'source': '京都市内の自宅・知人宅', 'target': '祇園方面', 'value': 71},
        {'source': 'ＪＲ京都駅（在来線）', 'target': '祇園方面', 'value': 223},
        {'source': '京都市内の宿泊施設', 'target': '祇園方面', 'value': 256},
        {'source': '河原町・新京極方面', 'target': 'ＪＲ京都駅（在来線）', 'value': 4},
        {'source': '京阪駅', 'target': '祇園方面', 'value': 95},
        {'source': '河原町・新京極方面', 'target': '下鴨神社周辺', 'value': 12},
        {'source': '下鴨神社周辺', 'target': '平安神宮周辺', 'value': 2},
        {'source': '平安神宮周辺', 'target': '京阪駅', 'value': 14},
        {'source': '京阪駅', 'target': '平安神宮周辺', 'value': 3},
        {'source': '河原町・新京極方面', 'target': '祇園方面', 'value': 33},
        {'source': 'その他ＪＲ駅', 'target': '大原・八瀬方面', 'value': 1},
        {'source': '大原・八瀬方面', 'target': '河原町・新京極方面', 'value': 2},
        {'source': '河原町・新京極方面', 'target': '京阪駅', 'value': 51},
        {'source': '祇園方面', 'target': '平安神宮周辺', 'value': 25},
        {'source': '祇園方面', 'target': 'その他ＪＲ駅', 'value': 4},
        {'source': '祇園方面', 'target': '下鴨神社周辺', 'value': 14},
        {'source': '下鴨神社周辺', 'target': 'ＪＲ京都駅（新幹線）', 'value': 2},
        {'source': '祇園方面', 'target': '大原・八瀬方面', 'value': 6},
        {'source': '大原・八瀬方面', 'target': '京都市内の自宅・知人宅', 'value': 1},
        {'source': '祇園方面', 'target': '京阪駅', 'value': 10},
        {'source': '下鴨神社周辺', 'target': '京阪駅', 'value': 4},
        {'source': '平安神宮周辺', 'target': '河原町・新京極方面', 'value': 21},
        {'source': 'その他ＪＲ駅', 'target': '祇園方面', 'value': 28},
        {'source': '河原町・新京極方面', 'target': '京都市内の宿泊施設', 'value': 13},
        {'source': '平安神宮周辺', 'target': '下鴨神社周辺', 'value': 8},
        {'source': '大原・八瀬方面', 'target': '京都市内の宿泊施設', 'value': 1},
        {'source': '祇園方面', 'target': '京都市内の自宅・知人宅', 'value': 3},
        {'source': '河原町・新京極方面', 'target': '平安神宮周辺', 'value': 6},
        {'source': '京都市内の自宅・知人宅', 'target': '河原町・新京極方面', 'value': 7},
        {'source': '河原町・新京極方面', 'target': '京都市内の自宅・知人宅', 'value': 6},
        {'source': '大原・八瀬方面', 'target': 'ＪＲ京都駅（在来線）', 'value': 1},
        {'source': '下鴨神社周辺', 'target': '河原町・新京極方面', 'value': 13},
        {'source': '祇園方面', 'target': 'ＪＲ京都駅（在来線）', 'value': 10},
        {'source': '京阪駅', 'target': '下鴨神社周辺', 'value': 1},
        {'source': 'ＪＲ京都駅（新幹線）', 'target': '河原町・新京極方面', 'value': 2},
        {'source': '祇園方面', 'target': 'ＪＲ京都駅（新幹線）', 'value': 12},
        {'source': '京都市内の宿泊施設', 'target': '河原町・新京極方面', 'value': 13},
        {'source': '大原・八瀬方面', 'target': '下鴨神社周辺', 'value': 2},
        {'source': '大原・八瀬方面', 'target': 'その他ＪＲ駅', 'value': 1},
        {'source': 'その他ＪＲ駅', 'target': '河原町・新京極方面', 'value': 1},
        {'source': '祇園方面', 'target': '京都市内の宿泊施設', 'value': 1},
        {'source': 'ＪＲ京都駅（在来線）', 'target': '河原町・新京極方面', 'value': 1},
        {'source': '下鴨神社周辺', 'target': '大原・八瀬方面', 'value': 1},
        {'source': '大原・八瀬方面', 'target': '京阪駅', 'value': 1},
        {'source': '京阪駅', 'target': '河原町・新京極方面', 'value': 3},
        {'source': '平安神宮周辺', 'target': '大原・八瀬方面', 'value': 1},
        {'source': '下鴨神社周辺', 'target': '祇園方面', 'value': 1},
        {'source': '下鴨神社周辺', 'target': 'ＪＲ京都駅（在来線）', 'value': 1},
        {'source': '河原町・新京極方面', 'target': '大原・八瀬方面', 'value': 1},
        {'source': '大原・八瀬方面', 'target': 'ＪＲ京都駅（新幹線）', 'value': 1},
        {'source': '平安神宮周辺', 'target': 'ＪＲ京都駅（新幹線）', 'value': 1},
        {'source': '下鴨神社周辺', 'target': 'その他ＪＲ駅', 'value': 1},
        {'source': '大原・八瀬方面', 'target': '祇園方面', 'value': 1}
    ]

    
    
    """
    """
    def sankey_echart(trip_chain_df, target_node_idx, place_labels_jp):
    node_num = 37  # totally 37 areas in the survey
    if len(str(target_node_idx)) == 0:
        raise ValueError('Please input valid node index!')
    if target_node_idx > node_num - 1:
        raise IndexError('Node index exceeds the total number of attraction areas!')

    nodes_involve_idx = set()
    # first loop: append nodes that are included in trips consist of the target node
    for _trip in trip_chain_df:
        if target_node_idx in _trip:
            for _pointer in range(len(_trip) - 1):
                _from, _to = _trip[_pointer], _trip[_pointer + 1]
                if _from == target_node_idx or _to == target_node_idx:
                    nodes_involve_idx.update((_from, _to))  # update both origin and dest.

    # generate nodes data structure for Sankey plot input
    _nodes = []
    for _ in nodes_involve_idx:
        _nodes.append({'name': place_labels_jp[_] + '_出発'})
        _nodes.append({'name': place_labels_jp[_] + '_到着'})

    # the second loop: enumerate all trips with nodes involved. Sum up the trip frequency
    edge_involve_dict = {}
    # dict: key: str(source) + '-' + str(target) in indices
    #        value: dict: key: origin, destination, frequency

    for _trip in trip_chain_df:
        for _pointer in range(len(_trip) - 1):
            _from, _to = _trip[_pointer], _trip[_pointer + 1]
            # only for the trips whose o and d are both involved
            if _from in nodes_involve_idx and _to in nodes_involve_idx:
                edge_pattern = str(_from) + '-' + str(_to)
                if edge_pattern in edge_involve_dict:
                    edge_involve_dict[edge_pattern]['value'] += 1
                else:
                    edge_involve_dict[edge_pattern] = {'source': _from, 'target': _to, 'value': 1}

    # generate the from and to dicts, sort the flows and merge the rest into "Others"
    x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
    sorted_x = {k: v for k, v in sorted(x.items(), key=lambda item: item[1])}


    # generate link data structure for Sankey plot input
    _links = []
    for k, v in edge_involve_dict.items():
        # avoid within node trips
        if v['source'] != v['target']:
            _links.append(
                {'source': place_labels_jp[v['source']] + '_出発',
                 'target': place_labels_jp[v['target']] + '_到着',
                 'value': v['value']}
            )

    _res = (
        Sankey().add(
            'Number of trips',
            _nodes,
            _links,
            linestyle_opt=opts.LineStyleOpts(opacity=0.2, curve=0.5, color="source", type_="dotted"),
            label_opts=opts.LabelOpts(position="right", ),
        ).set_global_opts(title_opts=opts.TitleOpts(title="A Sankey Diagram of Node {}".format(target_node_idx + 1)))
    )
    # output visualization by html file
    _res.render('result.html')

    # def sankey_for_attraction(trip_chain_df, target_node_idx, place_labels_jp):
    # # Input: normalized node index, starting from 0, i.e node_index in survey -1.
    # node_num = 37  # totally 37 areas in the survey
    # if len(str(target_node_idx)) == 0:
    #     raise ValueError('Please input valid node index!')
    # if target_node_idx > node_num - 1:
    #     raise IndexError('Node index exceeds the total number of attraction areas!')
    # 
    # nodes_involve_idx = set()
    # # first loop: append nodes that are included in trips consist of the target node
    # for _trip in trip_chain_df:
    #     if target_node_idx in _trip:
    #         for _pointer in range(len(_trip) - 1):
    #             _from, _to = _trip[_pointer], _trip[_pointer + 1]
    #             if _from == target_node_idx or _to == target_node_idx:
    #                 nodes_involve_idx.update((_from, _to))  # update both origin and dest.
    # 
    # # generate nodes data structure for Sankey plot input
    # _nodes = []
    # for _ in nodes_involve_idx:
    #     _nodes.append({'name': place_labels_jp[_] + '_出発'})
    #     _nodes.append({'name': place_labels_jp[_] + '_到着'})
    # 
    # # the second loop: enumerate all trips with nodes involved. Sum up the trip frequency
    # edge_involve_dict = {}
    # # dict: key: str(source) + '-' + str(target) in indices
    # #        value: dict: key: origin, destination, frequency
    # 
    # for _trip in trip_chain_df:
    #     for _pointer in range(len(_trip) - 1):
    #         _from, _to = _trip[_pointer], _trip[_pointer + 1]
    #         # only for the trips whose o and d are both involved
    #         if _from in nodes_involve_idx and _to in nodes_involve_idx:
    #             edge_pattern = str(_from) + '-' + str(_to)
    #             if edge_pattern in edge_involve_dict:
    #                 edge_involve_dict[edge_pattern]['value'] += 1
    #             else:
    #                 edge_involve_dict[edge_pattern] = {'source': _from, 'target': _to, 'value': 1}
    # 
    # _node_label = list(nodes_involved)
    # 
    # # generate link data structure for Sankey plot input
    # source, target, value = [], [], []
    # for k, v in edge_involve_dict.items():
    #     # avoid within node trips
    #     # if v['source'] != v['target']:
    #     source.append(_node_label.index(v['source']))
    #     target.append(_node_label.index(v['target']))
    #     value.append(v['value'])
    # 
    # # generate link data structure for Sankey plot input
    # fig = go.Figure(data=[go.Sankey(
    #     valueformat=".0f",
    #     node=dict(
    #         pad=15,
    #         thickness=15,
    #         line=dict(color="black", width=0.5),
    #         label=_node_label,
    #     ),
    #     link=dict(
    #         source=source,  # indices correspond to labels, eg A1, A2, A2, B1, ...
    #         target=target,
    #         value=value
    #     ))])
    # 
    # fig.update_layout(title_text="Trip flow Sankey Diagram for attraction {}".format(
    #     place_labels_jp[target_node_idx]
    # ), font_size=10)
    # 
    # fig.show()


    
    """
