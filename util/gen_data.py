import math
import statistics
import numpy as np

from datetime import datetime
from sklearn.cluster import KMeans
from util.ts_characteristics import cal_cosine_distance_of_ts


def gen_data(args, data):
    """
    One day contains 288 time steps. We calculate median of each step at different days.
    Never ever reach Test Set! We only work on 80% of the whole series.
    """
    volume = data[..., 0:1]
    total_length = volume.shape[0]
    cycle = args.daily_time_step  # 288
    idx_1 = int(total_length * args.train_ratio)  # 0.7
    idx_2 = int(total_length * args.val_ratio) + idx_1  # 0.9

    begin_date = args.date_range[0:10]  # eg. "2016/07/01" string length is 10
    begin_weekday = datetime.strptime(begin_date, '%Y/%m/%d').weekday()  # range from 0(Mon) to 6(Sat)
    # cal_cosine_distance_of_ts(volume, begin_weekday)

    # Specify the metrics to calculate the median. For example,
    # [[0, 1, 2, 3], [4], [5], [6]] means Mon-Thur, Fri, Sat, Sun
    # [[0, 1, 2], [3], [4], [5, 6]] means Mon-Wed, Thur, Fri, Sat-Sun

    # Construct cluster
    clusters = [[0, 1, 2, 3], [4], [5], [6]]
    cluster_list = []
    for _ in range(len(clusters)):  # 4
        hour_list = []
        for _ in range(24):
            hour_list.append([])
        cluster_list.append(hour_list)

    # Fill the cluster
    ts = volume[:idx_2]
    hourly_time_step = args.daily_time_step // 24  # 12 steps per hour
    step = 0
    while step < ts.shape[0]:  # <20000+
        weekday = step // args.daily_time_step % 7  # get Weekday
        cluster = -1
        for i in range(len(clusters)):
            if weekday in clusters[i]:
                cluster = i  # get cluster
                break
        hour = step % args.daily_time_step // hourly_time_step  # get hour
        cluster_list[cluster][hour].append(ts[step: step+hourly_time_step])
        step += hourly_time_step

    # Calculate median values
    for c in range(len(clusters)):
        for h in range(24):
            tmp = np.concatenate(cluster_list[c][h], axis=0)  # (12s, N)  s equals to 7 or 8 in PeMS08 WHEN the cluster contains only one day!
            tmp = np.median(tmp, axis=0)  # (N,)
            cluster_list[c][h] = tmp

    # reform to MTS
    tmp_list = []
    for c in range(len(clusters)):
        tmp_list.append(np.repeat(np.concatenate(cluster_list[c], axis=-1), 12, axis=1))  # 24 --(x12)--> 288 because 1 hour contains 12 steps
        tmp_list[c] = np.tile(tmp_list[c], (1, len(clusters[c])))  # repeat N times. N is cluster size.
    weekly = np.concatenate(tmp_list, axis=-1).transpose((1, 0))
    mid_series = np.tile(weekly, (total_length // cycle + 1, 1))[:total_length]
    mid_series = np.expand_dims(mid_series, axis=-1)
    gap_series = volume - mid_series
    gap_series += mid_series

    mid_series = np.concatenate([mid_series, data[..., 1:2], data[..., 2:3]], axis=-1)
    gap_series = np.concatenate([gap_series, data[..., 1:2], data[..., 2:3]], axis=-1)

    return np.concatenate([gap_series, mid_series[..., 0:1]], axis=-1), begin_weekday  # (T, N, )

    # clusters = [[0, 1, 2, 3], [4], [5], [6]]
    # clustered_volume = [[] for _ in range(7)]
    # packed_volume = []
    # median_volume = []
    # for i in range(len(volume) // args.daily_time_step):
    #     weekday = (i + begin_weekday) % 7
    #     begin_idx = i * args.daily_time_step
    #     clustered_volume[weekday].append(volume[begin_idx: begin_idx+args.daily_time_step])
    # for weekday in range(7):
    #     pack = np.concatenate(clustered_volume[weekday], axis=-1)
    #     packed_volume.append(pack)
    # for weekday in range(7):
    #     weekdays = find_elem_in_same_clusters(weekday, clusters)  # idx list
    #     daily_list = [packed_volume[idx] for idx in weekdays]
    #     daily = np.concatenate(daily_list, axis=-1)  # n(288, N, s) -> (288, N, ns)
    #     median = np.median(daily, axis=-1)  # (288, N) after calculating the median
    #     median_volume.append(median)
    # weekly = np.concatenate(median_volume, axis=0)
    # median = np.tile(weekly, (total_length // (cycle * 7) + 1, 1))[:total_length]
    # gap = volume[..., 0] - median
    #
    # mid_series = np.stack([median, data[..., 1], data[..., 2]], axis=-1)  # concat s-t idx
    # gap_series = np.stack([gap, data[..., 1], data[..., 2]], axis=-1)  # concat s-t idx
    # return np.concatenate([gap_series, mid_series[..., 0:1]], axis=-1), begin_weekday


def find_elem_in_same_clusters(weekday, clusters):
    for c in clusters:
        if weekday in c:
            return c
