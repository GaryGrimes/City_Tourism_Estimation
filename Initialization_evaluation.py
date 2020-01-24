import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pandas as pd
import os
import pylab as pl

# 设置matplotlib正常显示中文和负号
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# data = pd.read_excel(
# os.path.join(os.path.dirname(__file__), 'Evaluation result', 'Initialization evaluation result.xlsx'))
# data = pd.read_excel(os.path.join(os.path.dirname(__file__), 'Evaluation result', 'Initialization_result_2019-10-21.xlsx'))
# data = pd.read_excel(
#     os.path.join(os.path.dirname(__file__), 'Evaluation result',
#                  'Initialization result with new penalty function.xlsx'))


if __name__ == '__main__':
    # set filename to read
    initial_eval_filename = 'Initialization objective values ILS_Gamma.xlsx'  # generated on Jan. 19

    data = pd.read_excel(
        os.path.join(os.path.dirname(__file__), 'Evaluation result',
                     'Jan 20 grid search', initial_eval_filename), index_col=0)

    pnt = data.loc[:, 'penalty']

    x_min, x_max = min(pnt), max(pnt)

    plt.hist(pnt, bins=15, facecolor="darkblue", edgecolor="black", alpha=0.7)
    # 显示横轴标签
    plt.xlabel("Intervals")
    # 显示纵轴标签
    plt.ylabel("Frequency")
    # 显示图标题
    plt.title("Parameter Score distribution")
    plt.show()

    # number of parameters
    num_para = data.shape[1] - 3
    # a1的score distribution

    Stats = []

    for _i in range(num_para):
        param_name = data.columns[1 + _i]
        stat_temp = pd.DataFrame(columns=np.sort(data[param_name].unique()))
        for _idx, _value in enumerate(np.sort(data[param_name].unique())):
            stat_temp[stat_temp.columns[_idx]] = data['penalty'][data[param_name] == _value].values

        Stats.append(stat_temp)

    Stats_describe = [_.describe() for _ in Stats]

    # temp = {}
    # temp[]
    # a1, a2, b2, b3 = {}, {}, {}, {}
    #
    # for i in data['a1'].unique():
    #     a1[i] = pnt[data['a1'] == i]
    # for i in data['a2'].unique():
    #     a2[i] = pnt[data['a2'] == i]
    # for i in data['b2'].unique():
    #     b2[i] = pnt[data['b2'] == i]
    # for i in data['b3'].unique():
    #     b3[i] = pnt[data['b3'] == i]
    #
    # # for i in data['b3'].unique():
    # #     b3[i] = pnt[data['b3'] == i]
    #
    # val_dicts = [a1, a2, b2, b3]
    # names = ['a1', 'a2', 'b2', 'b3']
    #
    # # empty DFs
    # stat_a1 = pd.DataFrame(columns=np.sort(data['a1'].unique()))
    # stat_a2 = pd.DataFrame(columns=np.sort(data['a2'].unique()))
    # stat_b2 = pd.DataFrame(columns=np.sort(data['b2'].unique()))
    # stat_b3 = pd.DataFrame(columns=np.sort(data['b3'].unique()))
    #
    # stats = [stat_a1, stat_a2, stat_b2, stat_b3]
    # #
    # for _idx, _ in enumerate(stats):
    #     for i in range(_.shape[1]):
    #         _[_.columns[i]] = val_dicts[_idx][data[names[_idx]].unique()[i]].values
    #
    # # describe
    # sta_res_a1 = stat_a1.describe()
    # sta_res_a2 = stat_a2.describe()
    # sta_res_b2 = stat_b2.describe()
    # sta_res_b3 = stat_b3.describe()
    #
    #
    # # sta_res_b3 = stat_b3.describe()
    #
    # # histogram
    # stat_a1.hist(range=(x_min, x_max), bins=15, grid=True, sharey=True, figsize=(15, 12))
    #
    # plt.title('a1')
    # plt.show()
    #
    # # 补全a2~a5的
    #
    # # stat_a2.hist(range=(16000, 20000), bins=20, grid=True, sharey=True, figsize=(15, 12))
    # stat_a2.hist(range=(x_min, x_max), bins=15, grid=True, sharey=True, figsize=(15, 12))
    # plt.show()
    #
    # stat_b2.hist(range=(x_min, x_max), bins=15, grid=True, sharey=True, figsize=(15, 12))
    # plt.show()
    #
    # # stat_b3.hist(range=(x_min, x_max), bins=15, grid=True, sharey=True, figsize=(15, 12))
    # # plt.show()
    #
    # # #  做grid mesh，fix a1 to 1 and a2 to 0.03
    # #
    # # a1, a2 = 1, 0.03
    # #
    # # vector_b2, vector_b3 = data['b2'].unique(), data['b3'].unique()
    # #
    # # grid = pd.DataFrame(np.zeros([7, 7], dtype=float), columns=data['b2'].unique(), index=data['b2'].unique())
    # #
    # # for i in range(7):
    # #     for j in range(7):
    # #         grid.iloc[i, j] = data[(data['a1'] == 1) & (data['a2'] == 0.03) & (data['b2'] == data['b2'].unique()[i]) & (
    # #                 data['b3'] == data['b3'].unique()[j])]['penalty'].values
    # #
    # # # grid.to_excel('Meshgrid_b2_b3.xlsx')
