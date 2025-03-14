# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import warnings

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception as e:
    warnings.warn(f'Runing without matplotlib {e}')

color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
line_list = ['-', '--', '-.', ':']


def plot_multi_curves(x,
                      y,
                      show=False,
                      title=None,
                      save_path=None,
                      x_label=None,
                      y_label=None):
    '''
    Args:
        x: the x-axis data
        y: the y-axis data dict
            like: [{"data": np.ndarrays, "label": ""}]
        title: None
        show: False
        save_path: None
        x_label: None
        y_label: None
    Returns:
    '''
    if save_path is not None:
        plt.figure()

    x_max, x_min = np.max(x), np.min(x)

    max_num, min_num = 0, 0
    for y_id, data in enumerate(y):
        max_n = np.max(data['data'])
        min_n = np.min(data['data'])
        max_num = max_n if max_n > max_num else max_num
        min_num = min_n if min_n < min_num else min_num
        plt.plot(x,
                 data['data'],
                 linestyle=line_list[y_id % len(line_list)],
                 linewidth=2,
                 color=color_list[y_id % len(color_list)],
                 label=data['label'],
                 alpha=1.00)
        plt.title(title, loc='center')

    plt.legend(loc='upper right')
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)

    x_step = (x_max - x_min) / 5
    y_step = (max_num - min_num) / 5
    plt.xticks(np.arange(x_min - x_step / 2, x_max + x_step / 2, x_step))
    plt.yticks(np.arange(min_num - y_step / 2, max_num + y_step / 2, y_step))
    plt.grid()
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.clf()
    plt.cla()
    plt.close()
    return True


def plt_curve(x,
              y,
              show=False,
              title=None,
              save_path=None,
              x_label=None,
              y_label=None):
    '''
    Args:
        x: the x-axis data
        y: the y-axis data dict
            like: [{"data": np.ndarrays, "label": ""}]
        title: None
        show: False
        save_path: None
        x_label: None
        y_label: None
    Returns:
    '''
    return plot_multi_curves(x, [{
        'data': y,
        'label': 'y'
    }],
                             show=show,
                             title=title,
                             save_path=save_path,
                             x_label=x_label,
                             y_label=y_label)
