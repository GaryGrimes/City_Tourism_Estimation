""" This script is to parse the observed complaints about trips from survey and output statistics
which might be helpful for targeting congested OD pairs.
Reference like wordcloud generation could be found in 'slvr/DataWrapping'
注意：该scipt在parse trips的时候并没有用到slvr package内的DataWrapping产生的trip database，因为simulation的结果也是不计算
intrazonal movements的，所以在仿真系统内的Data来说，统一将不考虑TAZ内trips；而本script目的只是parse statistics，所以trip的定义、
以及raw data会不同。这里没有用到trip database."""

import pandas as pd
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import progressbar as pb
from wordcloud import WordCloud
from matplotlib import rcParams


def comp_extractor(_arr):
    for x in _arr:
        _row, _col = x // 10 - 1, x % 10 - 1
        yield comp_options[_row][_col]


rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic',
                               'Noto Sans CJK JP']

comp_options = [
    ['道路が混雑', '経路がわかりにくい', '道路が細い', '駐車場', '駐車の待ち時間', '駐車代'],
    ['バスの本数', '運賃', '道路が混雑', '乗り換え', 'バスの選択', 'バス停わかりにくい', 'バス運転', '車内で混雑'],
    ['電車の本数', '鉄道運賃', '乗り換え', '駅のバリアフリー', '駅がわからない', '電車内で混雑', '車内で混雑', '乗り換えが面倒']
]

if __name__ == '__main__':
    # %% DATA PREPARATION
    # read OD data
    print('Reading OD data...\n')
    OD_data = pd.read_excel(os.path.join(os.path.dirname(__file__), 'slvr', 'Database', '観光客の動向調査.xlsx'),
                            sheet_name='OD')
    OD_data[['飲食費', '土産代']] = OD_data[['飲食費', '土産代']].replace(np.nan, 0).astype(int)

    # generate an array with trip od and index
    trip_od = OD_data.loc[:, ['出発地', '到着地']].values

    # %% complaints and dissatisfaction
    comp_freq_table = OD_data.loc[:, '不満点１':'不満点６'].values
    temp = comp_freq_table.flatten()
    comp_fre_array = temp[temp > 0].astype(int)

    y = np.bincount(comp_fre_array)
    ii = np.nonzero(y)[0]

    com_fre = list(zip(ii, y[ii]))

    wordcloud_flag = input("Generate wordcloud? Press 'Enter' to skip")

    if wordcloud_flag:
        # Generate wordcloud txt
        g = comp_extractor(comp_fre_array)

        with open(os.path.join(os.path.dirname(__file__), 'slvr', 'Database', 'comp_output.txt'), 'w') as f:
            for x in g:
                f.write(str(x) + ' ')

        with open(os.path.join(os.path.dirname(__file__), 'slvr', 'Database', 'comp_output.txt'), 'r') as f:
            word_text = f.read()

        # create word-cloud;
        font_path = "/System/Library/fonts/NotoSansCJKjp-Regular.otf"
        wordcloud = WordCloud(font_path=font_path, regexp="[\w']+", background_color=None, mode='RGBA', scale=2,
                              colormap='magma')
        wordcloud.generate(word_text)

        # plot
        plt.figure(figsize=(20, 10))
        plt.imshow(wordcloud, interpolation='bilinear')

        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.show()

    # similarly, bus one-day bus 不使用的理由之类的, 也可以同样做一个wordcloud.
    pass

    # %% parse the indices of trips of certain complaints
    comp_options = [
        ['道路が混雑', '経路がわかりにくい', '道路が細い', '駐車場', '駐車の待ち時間', '駐車代'],
        ['バスの本数', '運賃', '道路が混雑', '乗り換え', 'バスの選択', 'バス停わかりにくい', 'バス運転', '車内で混雑'],
        ['電車の本数', '鉄道運賃', '乗り換え', '駅のバリアフリー', '駅がわからない', '電車内で混雑', '車内で混雑', '乗り換えが面倒']
    ]
    # in survey data rows and cols start from 1
    # congestion
    congestion_comp = [11, 23, 28]  # 1: car, 2: bus
    congestion_trip_idx = []

    # cabin congestion
    c_congestion_comp = [28, 36, 37]
    c_congestion_trip_idx = []

    # transit line and frequency
    line_freq_comp = [21, 31]
    line_freq_trip_idx = []

    # transit fare
    fare_comp = [22, 32]
    fare_trip_idx = []

    # enumerate comp_freq_table and parse trip indices with certain complaints
    for _idx, _comp in enumerate(comp_freq_table):
        for _ in _comp:
            if _ in congestion_comp:
                congestion_trip_idx.append(_idx)
            elif _ in c_congestion_comp:
                c_congestion_trip_idx.append(_idx)
            elif _ in line_freq_comp:
                line_freq_trip_idx.append(_idx)
            elif _ in fare_comp:
                fare_trip_idx.append(_idx)
        pass

    # avoid duplicates
    congestion_trip_idx, c_congestion_trip_idx, line_freq_trip_idx, fare_trip_idx = \
        set(congestion_trip_idx), set(c_congestion_trip_idx), set(line_freq_trip_idx), set(fare_trip_idx)

    # %% create complaints matrix and heatmap
    # valid zone numbers 1-47

    # inter_attractions 1-37
    congestion_comp_mat = np.zeros([37, 37])
    for _idx in congestion_trip_idx:
        try:
            congestion_comp_mat[trip_od[_idx][0], trip_od[_idx][1]] += 1
        except IndexError:
            continue

    c_congestion_comp_mat = np.zeros([37, 37])
    for _idx in c_congestion_trip_idx:
        try:
            c_congestion_comp_mat[trip_od[_idx][0], trip_od[_idx][1]] += 1
        except IndexError:
            continue

    line_freq_comp_mat = np.zeros([37, 37])
    for _idx in line_freq_trip_idx:
        try:
            line_freq_comp_mat[trip_od[_idx][0], trip_od[_idx][1]] += 1
        except IndexError:
            continue

    fare_comp_mat = np.zeros([37, 37])
    for _idx in fare_trip_idx:
        try:
            fare_comp_mat[trip_od[_idx][0], trip_od[_idx][1]] += 1
        except IndexError:
            continue

    # %% todo 	Complaints data should reflect the percentage (complaints/total trips),
    #  todo and

    # parse observed trip frequency table. From OD data not agents.pickle file

    # only parse trips between attractions 1-37

    observed_trip_table = np.zeros([37, 37], dtype=int)
    _os, _ds = OD_data['出発地'].values - 1, OD_data['到着地'].values - 1  # index in observed trip tables starts from 0
    for _idx in range(len(_os)):
        try:
            observed_trip_table[_os[_idx], _ds[_idx]] += 1
        except IndexError:
            continue
    df_observed_trip_table = pd.DataFrame(observed_trip_table, dtype=int)
    flag = 0
    if flag:
        df_observed_trip_table.to_excel('Project Database/Trips/Observed_trip_table.xlsx')

    per_congestion_comp_mat = congestion_comp_mat / observed_trip_table
    per_c_congestion_comp_mat = c_congestion_comp_mat / observed_trip_table
    per_line_freq_comp_mat = line_freq_comp_mat / observed_trip_table
    per_fare_comp_mat = fare_comp_mat / observed_trip_table

    # filter out very sparse OD pairs, i.e. <= 10 or 15…
    for i in range(observed_trip_table.shape[0]):
        for j in range(observed_trip_table.shape[1]):
            if observed_trip_table[i, j] <= 15:
                per_congestion_comp_mat[i, j] = 0
                per_c_congestion_comp_mat[i, j] = 0
                per_line_freq_comp_mat[i, j] = 0
                per_fare_comp_mat[i, j] = 0
    flag = 0
    if flag:
        pd.DataFrame(100 * per_congestion_comp_mat).round(1).to_excel(
            'Project Database/Complaints statistics/congestion_comp_rate.xlsx')
        pd.DataFrame(100 * per_c_congestion_comp_mat).round(1).to_excel(
            'Project Database/Complaints statistics/cabin_congestion_comp_rate.xlsx')

        pd.DataFrame(100 * per_line_freq_comp_mat).round(1).to_excel(
            'Project Database/Complaints statistics/line_freq_comp_rate.xlsx')
        pd.DataFrame(100 * per_fare_comp_mat).round(1).to_excel(
            'Project Database/Complaints statistics/fare_comp_rate.xlsx')

    # %% write data?
    flag_write = input("Write complaints data to excel? Press 'Enter' to skip, any key to continue.")
    if flag_write:
        temp_df = pd.DataFrame(congestion_comp_mat)
        temp_df.to_excel('Project Database/Complaints statistics/congestion complaints.xlsx',
                         sheet_name='congestion')

        temp_df = pd.DataFrame(c_congestion_comp_mat)
        temp_df.to_excel('Project Database/Complaints statistics/cabin congestion complaints.xlsx',
                         sheet_name='c_congestion')

        temp_df = pd.DataFrame(line_freq_comp_mat)
        temp_df.to_excel('Project Database/Complaints statistics/line freq complaints.xlsx',
                         sheet_name='line_freq')

        temp_df = pd.DataFrame(fare_comp_mat)
        temp_df.to_excel('Project Database/Complaints statistics/fare complaints.xlsx',
                         sheet_name='fare')
