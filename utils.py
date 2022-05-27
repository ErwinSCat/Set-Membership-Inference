import heapq
import sys
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

# classes
class PriorityQueue:
    def __init__(self):
        self._queue = list()

    def push(self, priority, item):
        heapq.heappush(self._queue, (-priority, item))

    def pop(self):
        return heapq.heappop(self._queue)[-1]

    def is_empty(self):
        return len(self._queue) == 0


class Logger:
    def __init__(self, dir_name='./log/', file_name='log.txt'):
        self.log_file = dir_name + file_name
        self.terminal = sys.stdout
        self.log = None

    def write(self, message):
        self.terminal.write(message)
        try:
            self.log = open(self.log_file, 'a')
            self.log.write(message)
            self.log.close()
        except Exception as err:
            pass

    def flush(self):
        pass


# functions
def from_log_points_to_time_series(log_points):
    prev_index = 0
    prev_num = 0
    time_series = list()
    for log_point in log_points:
        assert log_point[0] >= prev_index
        if log_point[0] > prev_index:
            for i in range(log_point[0] - prev_index - 1):
                time_series.append(prev_num)
            time_series.append(log_point[1])
        else:
            time_series[-1] = log_point[1]

        prev_index = log_point[0]
        prev_num = log_point[1]

    return time_series


def calculate_mutual_information(df_featured_population, features, victim_set):
    df_featured_population['set_membership'] = 0
    df_featured_population.loc[
        df_featured_population['user'].isin(victim_set),
        ['set_membership']
    ] = 1

    data = np.array(df_featured_population[features])
    label = np.array(df_featured_population['set_membership'])

    MI_values = mutual_info_classif(data, label, discrete_features=True)
    MI_results = pd.DataFrame(columns=features, data=[MI_values])

    print('    - mutual information')
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 1000)
    print(MI_results)
