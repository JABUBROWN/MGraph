import math
import statistics
import numpy as np


def cal_cosine_distance_of_ts(series, begin_weekday, cycle=288*7):

    # cal hour average volume of different days
    # # [1] set up the map
    week_map = dict()  # 7 * 24 map with list() elements
    for week in range(7):
        for hour in range(24):
            week_map[(week, hour)] = list()

    # # [2] fill the map
    avg_t = np.average(series, axis=1)  # cal average volume of all nodes
    for week in range(7):
        for hour in range(24):
            begin = week * 288 + hour * 12
            while begin < series.shape[0]:  # begin < 20000+
                avg = avg_t[begin: begin + 12]
                week_map[(week, hour)].append(np.average(avg))  # append the average number to the list
                begin += cycle  # plus 2016

    # # [3] cal average of the map
    for week in range(7):
        for hour in range(24):
            week_map[(week, hour)] = statistics.mean(week_map[(week, hour)])

    # # [4] align week
    week_list = [[], [], [], [], [], [], []]
    for w in range(7):
        real_week = (w + begin_weekday) % 7
        for d in range(24):
            week_list[real_week].append(week_map[(w, d)])

    # # [5] cal cosine distance
    cos_dis_map = {}
    for w1 in range(7):
        for w2 in range(w1 + 1, 7):
            dist = cosine_distance(week_list[w1], week_list[w2])
            cos_dis_map[(w1, w2)] = dist

    # Print cosine distances matrix from Monday to Sunday
    print(cos_dis_map)


def cosine_distance(v1, v2):
    # 计算两个向量的点积
    dot_product = sum(v1[i] * v2[i] for i in range(len(v1)))

    # 计算两个向量的模
    norm_v1 = math.sqrt(sum(pow(v, 2) for v in v1))
    norm_v2 = math.sqrt(sum(pow(v, 2) for v in v2))

    # 计算余弦相似度
    cosine_similarity = dot_product / (norm_v1 * norm_v2)

    # 计算余弦距离
    cosine_distance = 1 - cosine_similarity
    return cosine_distance