from process import *
from attack import toy_attack, random_partition, kmeans_partition, DBSCAN_partition
from utils import Logger

import os
import sys


if __name__ == '__main__':
    rerun_flag = True
    dataset_name = 'covid_contact_tracing'
    dataset_params = {
        'covid_contact_tracing': {
            'dataset_name': 'covid_contact_tracing',
            'file_name': 'X size = {}, beta = {}, time_window = {}, start_round = {}, num. queries = {}',

            'attacker_set_size': -1,
            'num_iterations': 20,
            'partition_methods': [toy_attack, random_partition, kmeans_partition],
            'betas': dict(),
            'features': ['fever', 'cough', 'gender'],
            'time_windows': [14],
            'start_rounds': [14],
            'num_queries_per_victim_set': 5
        },
        'ad_conversion_revenue': {
            'dataset_name': 'ad_conversion_revenue',
            'file_name': 'X size = {}, beta = {}, time_window = {}, start_round = {}',

            'attacker_set_size': -1,
            'num_iterations': 20,
            'partition_methods': [toy_attack, random_partition, kmeans_partition],
            'betas': dict(),
            'features': ['final_gender_code', 'age_level', 'pvalue_level', 'shopping_level',
                         'occupation', 'new_user_class_level '],
            'time_windows': [8],
            'start_rounds': [3 * 24],
            'num_queries_per_victim_set': 1,

            'ad_id': 710164
        },
        'ad_conversion_lift': {
            'dataset_name': 'ad_conversion_lift',
            'file_name': 'X size = {}, beta = {}, time_window = {}, start_round = {}',

            'attacker_set_size': -1,
            'num_iterations': 20,
            'partition_methods': [toy_attack, random_partition, kmeans_partition],
            'betas': dict(),
            'features': ['age', 'gender', 'education', 'consumption_ability', 'marriage_status', 'work'],
            'time_windows': [32],
            'start_rounds': [3],
            'num_queries_per_victim_set': 1,

            'ad_id': 320379
        }
    }
    dataset_proc_func = {
        'covid_contact_tracing': process_dynamic_dataset,
        'ad_conversion_revenue': process_dynamic_dataset,
        'ad_conversion_lift': process_dynamic_dataset
    }

    # test whether plot data exists
    plot_data_dir = './plot_data/{}/'.format(dataset_name)
    exist_flag = os.path.isdir(plot_data_dir) and (len(os.listdir(plot_data_dir)) != 0)

    # attacker' set size array
    attacker_set_sizes = [512, 1024, 2048]

    # run attack
    if exist_flag is False or rerun_flag is True:
        # log file
        log_dir = './log/{}/'.format(dataset_name)
        os.makedirs(log_dir, exist_ok=True)
        sys.stdout = Logger(dir_name=log_dir, file_name='X sizes = {}, start rounds = {}, {}.txt'.format(
            attacker_set_sizes,
            dataset_params[dataset_name]['start_rounds'],
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S').replace(':', '_')
        ))

        for attacker_set_size in attacker_set_sizes:
            dataset_params[dataset_name]['attacker_set_size'] = attacker_set_size

            dataset_proc_func[dataset_name](dataset_params[dataset_name])

            print('[*] done')
            print()
    else:
        print('[*] plot data exists and rerun flag is not set')
