""" Defines boundaries for attraction areas and calculate area centers"""

import numpy as np
import pandas as pd
import os


if __name__ == '__main__':
    size = [37,5]
    Bndy = pd.DataFrame(np.zeros(size), dtype=float)

    boundaries = {1: [[135.8074, 35.1265], [135.8418, 35.0635]],
                      2: [[135.753141, 35.126590], [135.775803, 35.104627]],
                      3: [[135.765895, 35.083676], [135.790349, 35.056074]],
                      4: [[135.7478, 35.0631], [135.7602, 35.0529]],
                      5: [[135.6661, 35.0624], [135.6839, 35.0519]],
                      6: [[135.7922, 35.0561], [135.8099, 35.0430]],
                      7: [[135.726436, 35.056776], [135.734913, 35.050022]],
                      8: [[135.7569, 35.0536], [135.7705, 35.0443]],
                      9: [[135.7421, 35.0471], [135.7522, 35.0404]],
                      10: [[135.7257, 35.0422], [135.7333, 35.0369]],
                      11: [[135.7670, 35.0412], [135.7785, 35.0289]],
                      12: [[135.7306, 35.0352], [135.7429, 35.0269]],
                      13: [[135.7092, 35.0365], [135.7284, 35.0274]],
                      14: [[135.6625, 35.0306], [135.6924, 35.0191]],
                      15: [[135.79714, 35.02816], [135.80012, 35.02570]],
                      16: [[135.7899, 35.0254], [135.7985, 35.0095]],
                      17: [[135.7778, 35.0233], [135.7901, 35.0105]],
                      18: [[135.7591, 35.0345], [135.7675, 35.0175]],
                      19: [[135.7156, 35.0259], [135.7302, 35.0189]],
                      20: [[135.7426, 35.0173], [135.7552, 35.0113]],
                      21: [[135.7362, 35.0144], [135.7426, 35.0085]],
                      22: [[135.7043, 35.0175], [135.7142, 35.0132]],
                      23: [[135.6651, 35.0198], [135.6832, 35.0095]],
                      24: [[135.7724, 35.0092], [135.7863, 34.9983]],
                      25: [[135.7599, 35.0105], [135.7713, 34.9997]],
                      26: [[135.6791, 35.0019], [135.6900, 34.9901]],
                      27: [[135.7755, 34.9967], [135.7857, 34.9920]],
                      28: [[135.7655, 34.9923], [135.7765, 34.9860]],
                      29: [[135.7461, 34.9956], [135.7635, 34.9830]],
                      # the area was extended to include the Kyoto station bldg (shopping and gourmet)
                      30: [[135.7067, 34.9863], [135.7138, 34.9808]],
                      31: [[135.7694, 34.9849], [135.7822, 34.9748]],
                      32: [[135.7424, 34.9842], [135.7506, 34.9779]],
                      33: [[135.7688, 34.9723], [135.7837, 34.9627]],
                      34: [[135.8163, 34.9538], [135.8250, 34.9484]],
                      35: [[135.74553, 34.95167], [135.74933, 34.94878]],
                      36: [[135.7534, 34.9337], [135.7626, 34.9289]],
                      37: [[135.6076, 35.2130], [135.7253, 35.1223]]
                      }

    Bndy[0] = range(1,38)
    # 左上边界
    Bndy[1] = [boundaries[i][0][0] for i in range(1,38)]
    Bndy[2] = [boundaries[i][0][1] for i in range(1,38)]
    Bndy[3] = [boundaries[i][1][0] for i in range(1,38)]
    Bndy[4] = [boundaries[i][1][1] for i in range(1,38)]

    _ = 'boundaries' + '.csv'

    Bndy.to_csv(os.path.join(os.path.dirname(__file__), _))