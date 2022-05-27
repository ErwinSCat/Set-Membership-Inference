import datetime
import numpy as np
import pandas as pd
from numpy import unique
from sklearn.utils import shuffle
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from math import ceil
import copy
from utils import calculate_mutual_information


def load_Israel_covid_dataset(attacker_set_size, num_iterations, betas, features, time_windows, start_rounds, others=None):
    # read file
    file_path = './datasets/Israel_covid/corona_tested_individuals_ver_006.english.csv'
    df = pd.read_csv(file_path)
    print('[*] load dataset from {}'.format(file_path))
    print('[*] # rows = {}'.format(df.shape[0]))

    # get additional parameter
    num_queries_per_victim_set = others['num_queries_per_victim_set']

    # preprocessing dataset
    selected_columns = ['test_date', 'corona_result']
    selected_columns.extend(features)
    df = df[selected_columns]
    df.replace('0', 0, inplace=True)
    df.replace('1', 1, inplace=True)
    df.replace('negative', 0, inplace=True)
    df.replace('positive', 1, inplace=True)
    df.replace('female', 0, inplace=True)
    df.replace('male', 1, inplace=True)
    df['test_date'] = df.apply(lambda x: datetime.date.fromisoformat(x['test_date']), axis=1)

    # dataset cleaning
    df = df.replace('other', np.NaN).replace('None', np.NaN).dropna()
    print('[*] # rows after cleaning = {}'.format(df.shape[0]))

    # assign unique identifier
    df['user'] = [i for i in range(df.shape[0])]

    # dataset statistics
    start_time = df['test_date'].min()
    end_time = df['test_date'].max()
    print('    - time range: from {} to {}'.format(start_time, end_time))
    num_allowed_rounds = (end_time - start_time).days + 1
    print('    - # allowed rounds = {}'.format(num_allowed_rounds))

    # dataset period
    period = list()
    for round_counter in range(0, num_allowed_rounds):
        for i in range(num_queries_per_victim_set):
            period.append(
                str(start_time + datetime.timedelta(days=round_counter)) + ' {}/{}'.format(i + 1, num_queries_per_victim_set)
            )

    # return values
    victim_set_lists = dict()       # store a list of victim's sets for each (time_window) tuple
                                    # the list length = num_allowed_rounds
    attacker_set_lists = dict()     # store a list of attacker's *featured* sets for each (beta, start_round, time_window) tuple
                                    # the list length = the number of repeated iterations

    print('[*] # valid records = {}'.format(df.shape[0]))
    for start_round in start_rounds:
        assert 1 <= start_round <= num_allowed_rounds
        attack_start_time = start_time + datetime.timedelta(days=start_round - 1)
        print('[*] processing dataset for attack started at {}'.format(attack_start_time))

        for time_window in time_windows:
            # generate (i) victim's dynamic set per day for each time_window
            #          (ii) the positive/negative records w.r.t. start_round
            print('    - set time window = {} day(s)'.format(time_window))
            victim_set_lists[time_window] = list()

            for round_counter in range(1, num_allowed_rounds + 1):
                prev_time = max(
                    start_time,
                    start_time + datetime.timedelta(days=round_counter - time_window)
                )
                curr_time = start_time + datetime.timedelta(days=round_counter - 1)

                # collect users within the time window
                users = set(np.array(
                    df.loc[
                        (df['corona_result'] == 1) & (prev_time <= df['test_date']) & (df['test_date'] <= curr_time),
                        ['user']
                    ]['user']
                ))
                print('        > from {} to {}: {} distinct infected patients'.format(
                    prev_time, curr_time, len(users)
                ))
                victim_set_lists[time_window].append(users)

            # set betas
            ratio = len(victim_set_lists[time_window][start_round - 1]) / df.shape[0]
            betas[start_round] = [
                round(ratio, 4), round(5 * ratio, 4), round(10 * ratio, 4)
            ]
            print('    - start round = {}, betas = {}'.format(start_round, betas[start_round]))

            # compute mutual information
            df_featured_population = copy.deepcopy(df)
            df_featured_population.loc[
                (df_featured_population['test_date'] > attack_start_time) & (df_featured_population['corona_result'] == 1),
                ['fever', 'cough']
            ] = [0, 0]
            calculate_mutual_information(
                df_featured_population=df_featured_population, features=features,
                victim_set=victim_set_lists[time_window][start_round - 1]
            )

            for beta in betas[start_round]:
                num_sampled_positives = ceil(beta * attacker_set_size)
                num_sampled_negatives = attacker_set_size - num_sampled_positives
                print('        > beta = {}, attacker\'s set size = {}'.format(beta, attacker_set_size))
                print('        > uniformly draw {} positives and {} negatives'.format(
                    num_sampled_positives, num_sampled_negatives
                ))

                attacker_set_lists[(beta, start_round, time_window)] = list()
                for iteration_counter in range(num_iterations):
                    # generate a sample attacker's set for each iteration
                    sampled_positives = df.loc[
                        df['user'].isin(victim_set_lists[time_window][start_round - 1])
                    ].sample(n=num_sampled_positives, replace=False)
                    sampled_negatives = df.loc[
                        ~df['user'].isin(victim_set_lists[time_window][start_round - 1])
                    ].sample(n=num_sampled_negatives, replace=False)

                    sampled_attacker_set = shuffle(sampled_positives.append(sampled_negatives))

                    # prevent early features
                    sampled_attacker_set = copy.deepcopy(sampled_attacker_set)
                    sampled_attacker_set.loc[
                        (sampled_attacker_set['test_date'] > attack_start_time) & (sampled_attacker_set['corona_result'] == 1),
                        ['fever', 'cough']
                    ] = [0, 0]

                    sampled_attacker_set.drop(['test_date', 'corona_result'], axis=1, inplace=True)
                    assert sampled_attacker_set.shape[0] == sampled_positives.shape[0] + sampled_negatives.shape[0]
                    assert sampled_attacker_set.isnull().sum().sum() == 0

                    attacker_set_lists[(beta, start_round, time_window)].append(sampled_attacker_set)

                print()

    return period, attacker_set_lists, victim_set_lists, set(unique(df['user']))


def load_taobao_dataset(attacker_set_size, num_iterations, betas, features, time_windows, start_rounds, others=None):
    # read file
    click_log_path = './datasets/taobao/raw_sample.csv'
    user_feature_path = './datasets/taobao/user_profile.csv'
    df_log = pd.read_csv(click_log_path)
    df_featured_user = pd.read_csv(user_feature_path)
    print('[*] load dataset from {}, {}'.format(click_log_path, user_feature_path))
    print('[*] # rows = {}'.format(df_log.shape[0]))

    # get additional parameter
    ad_id = others['ad_id']

    # preprocessing dataset
    df_featured_user.rename(columns={'userid': 'user'}, inplace=True)

    df_log = df_log[['time_stamp', 'user', 'adgroup_id', 'clk']]

    selected_columns = ['user']
    selected_columns.extend(features)
    df_featured_user = df_featured_user[selected_columns]

    # dataset cleaning
    df_featured_user.drop_duplicates(['user'], inplace=True)
    df_log = df_log[df_log['user'].isin(df_featured_user['user'])]
    assert len(set(np.array(df_log['user']))) == len(set(np.array(df_featured_user['user'])))
    print('[*] # rows after cleaning = {}'.format(df_log.shape[0]))

    # dataset statistics
    start_time = datetime.datetime(year=2017, month=5, day=6, hour=0, minute=0, second=0)
    end_time = datetime.datetime(year=2017, month=5, day=14, hour=0, minute=0, second=0)
    print('    - time range: from {} to {}'.format(start_time, end_time))
    num_allowed_rounds = (end_time - start_time).days * 24
    print('    - # allowed rounds = {}'.format(num_allowed_rounds))

    # dataset period
    period = list()
    for round_counter in range(1, num_allowed_rounds + 1):
        period.append(start_time + datetime.timedelta(hours=round_counter))

    # return values
    victim_set_lists = dict()       # store a list of victim's sets for each (time_window) tuple
                                    # the list length = num_allowed_rounds
    attacker_set_lists = dict()     # store a list of attacker's *featured* sets for each (beta, start_round, time_window) tuple
                                    # the list length = the number of repeated iterations

    print('[*] processing ad id = {}'.format(ad_id))
    print('[*] # valid records = {} for ad id = {}'.format(df_log['adgroup_id'].value_counts()[ad_id], ad_id))
    print('[*] # valid users = {}'.format(len(unique(df_log['user']))))

    for start_round in start_rounds:
        assert 1 <= start_round <= num_allowed_rounds
        print('[*] processing dataset for attack started at {}'.format(
            start_time + datetime.timedelta(hours=start_round)
        ))

        for time_window in time_windows:
            # generate (i) victim's dynamic set per hour for each ad_id and time_window
            #          (ii) the click/non-click records w.r.t. start_round
            print('    - set time window = {} day(s)'.format(time_window))
            victim_set_lists[time_window] = list()

            for round_counter in range(1, num_allowed_rounds + 1):
                prev_timestamp = max(
                    start_time.timestamp(),
                    (start_time + datetime.timedelta(hours=round_counter) - datetime.timedelta(days=time_window)).timestamp()
                )
                curr_timestamp = (start_time + datetime.timedelta(hours=round_counter)).timestamp()

                # collect users within the time window
                users = set(np.array(
                    df_log.loc[
                        (df_log['adgroup_id'] == ad_id) & (df_log['clk'] == 1) &
                        (prev_timestamp <= df_log['time_stamp']) & (df_log['time_stamp'] <= curr_timestamp),
                        ['user']
                    ]['user']
                ))
                print('        > from {} to {}: {} distinct clickers'.format(
                    datetime.datetime.fromtimestamp(prev_timestamp),
                    datetime.datetime.fromtimestamp(curr_timestamp),
                    len(users)
                ))
                victim_set_lists[time_window].append(users)

            # set betas
            ratio = len(victim_set_lists[time_window][start_round - 1]) / len(unique(df_log['user']))
            betas[start_round] = [
                round(ratio, 4), round(5 * ratio, 4), round(10 * ratio, 4)
            ]
            print('    - start round = {}, betas = {}'.format(start_round, betas[start_round]))

            # compute mutual information
            df_featured_population = copy.deepcopy(df_featured_user)
            df_featured_population.fillna(-1, inplace=True)
            calculate_mutual_information(
                df_featured_population=df_featured_population, features=features,
                victim_set=victim_set_lists[time_window][start_round - 1]
            )

            for beta in betas[start_round]:
                num_sampled_clickers = ceil(beta * attacker_set_size)
                num_sampled_non_clickers = attacker_set_size - num_sampled_clickers
                print('        > beta = {}, attacker\'s set size = {}'.format(beta, attacker_set_size))
                print('        > uniformly draw {} clickers and {} non-clickers'.format(
                    num_sampled_clickers, num_sampled_non_clickers
                ))

                attacker_set_lists[(beta, start_round, time_window)] = list()
                for iteration_counter in range(num_iterations):
                    # generate a sample attacker's set for each iteration
                    sampled_clickers = df_featured_user.loc[
                        df_featured_user['user'].isin(victim_set_lists[time_window][start_round - 1])
                    ].sample(n=num_sampled_clickers, replace=False)
                    sampled_non_clickers = df_featured_user.loc[
                        ~df_featured_user['user'].isin(victim_set_lists[time_window][start_round - 1])
                    ].sample(n=num_sampled_non_clickers, replace=False)

                    sampled_attacker_set = shuffle(sampled_clickers.append(sampled_non_clickers))
                    assert sampled_attacker_set.shape[0] == sampled_clickers.shape[0] + sampled_non_clickers.shape[0]
                    # print('        > {} NaN values in {}-th iteration, to be filled'.format(
                    #     sampled_attacker_set.isnull().sum().sum(), iteration_counter + 1
                    # ))

                    # fill NaN values
                    sampled_attacker_set.fillna(-1, inplace=True)
                    assert sampled_attacker_set.isnull().sum().sum() == 0

                    attacker_set_lists[(beta, start_round, time_window)].append(sampled_attacker_set)

                print()

    return period, attacker_set_lists, victim_set_lists, set(unique(df_log['user']))


def load_tencent_dataset(attacker_set_size, num_iterations, betas, features, time_windows, start_rounds, others=None):
    # read file
    click_log_path = './datasets/tencent/testA/imps_log/totalExposureLog.out'
    user_feature_path = './datasets/tencent/testA/user/user_data'
    df_log = pd.read_csv(click_log_path, sep='\t', header=None)
    df_featured_user = pd.read_csv(user_feature_path, sep='\t', header=None)
    print('[*] load dataset from {}, {}'.format(click_log_path, user_feature_path))
    print('[*] # rows = {}'.format(df_log.shape[0]))

    # get additional parameter
    ad_id = others['ad_id']

    # preprocessing dataset
    column_name_log = ['ad_request_id', 'timestamp', 'ad_pos_id', 'user', 'ad_id',
                       'ad_size', 'bid', 'pctr', 'quality_ecpm', 'total_ecpm']
    column_name_feature = ['user', 'age', 'gender', 'area', 'marriage_status',
                           'education', 'consumption_ability', 'device', 'work', 'network_type', 'behaviour']
    df_log.columns = column_name_log
    df_featured_user.columns = column_name_feature

    df_log = df_log[['timestamp', 'user', 'ad_id']]

    selected_columns = ['user']
    selected_columns.extend(features)
    df_featured_user = df_featured_user[selected_columns]

    if 'marriage_status' in features:
        # df_featured_user['marriage_status'] = df_featured_user.apply(lambda x: str.split(x['marriage_status'], ','), axis=1)
        # mlb = MultiLabelBinarizer()
        # mlb_transformed = mlb.fit_transform([df_featured_user.loc[i, 'marriage_status'] for i in range(df_featured_user.shape[0])])
        #
        # extended_column_name = ['marriage_status_' + name for name in mlb.classes_]
        # df_featured_user = pd.concat([df_featured_user, pd.DataFrame(mlb_transformed, columns=extended_column_name)], axis=1)
        # df_featured_user.drop(['marriage_status'], axis=1, inplace=True)

        df_featured_user['marriage_status'] = df_featured_user.apply(
            lambda x: str(sorted(list(map(int, str(x['marriage_status']).replace(' ', '').split(','))))), axis=1
        )
        enc = LabelEncoder()
        enc_transformed = enc.fit_transform([df_featured_user.loc[i, 'marriage_status'] for i in range(df_featured_user.shape[0])])
        df_featured_user['marriage_status'] = enc_transformed

    if 'work' in features:
        # df_featured_user['work'] = df_featured_user.apply(lambda x: str.split(x['work'], ','), axis=1)
        # mlb = MultiLabelBinarizer()
        # mlb_transformed = mlb.fit_transform([df_featured_user.loc[i, 'work'] for i in range(df_featured_user.shape[0])])
        #
        # extended_column_name = ['work_' + name for name in mlb.classes_]
        # df_featured_user = pd.concat([df_featured_user, pd.DataFrame(mlb_transformed, columns=extended_column_name)], axis=1)
        # df_featured_user.drop(['work'], axis=1, inplace=True)

        df_featured_user['work'] = df_featured_user.apply(
            lambda x: str(sorted(list(map(int, str(x['work']).replace(' ', '').split(','))))), axis=1
        )
        enc = LabelEncoder()
        enc_transformed = enc.fit_transform([df_featured_user.loc[i, 'work'] for i in range(df_featured_user.shape[0])])
        df_featured_user['work'] = enc_transformed

    # dataset cleaning
    df_featured_user.drop_duplicates(['user'], inplace=True)
    df_log.dropna(inplace=True)
    df_featured_user = df_featured_user.loc[df_featured_user['user'].isin(df_log['user'])]
    assert len(set(np.array(df_log['user']))) == len(set(np.array(df_featured_user['user'])))
    print('[*] # rows after cleaning = {}'.format(df_log.shape[0]))

    # dataset statistics
    start_time = datetime.datetime(year=2019, month=2, day=16, hour=0, minute=0, second=0)
    end_time = datetime.datetime(year=2019, month=3, day=20, hour=0, minute=0, second=0)
    print('    - time range: from {} to {}'.format(start_time, end_time))
    num_allowed_rounds = (end_time - start_time).days
    print('    - # allowed rounds = {}'.format(num_allowed_rounds))

    # dataset period
    period = list()
    for round_counter in range(1, num_allowed_rounds + 1):
        period.append(start_time + datetime.timedelta(days=round_counter))

    # return values
    victim_set_lists = dict()       # store a list of victim's sets for each (time_window) tuple
                                    # the list length = num_allowed_rounds
    attacker_set_lists = dict()     # store a list of attacker's *featured* sets for each (beta, start_round, time_window) tuple
                                    # the list length = the number of repeated iterations

    print('[*] processing ad id = {}'.format(ad_id))
    print('[*] # valid exposure records = {} for ad id = {}'.format(df_log['ad_id'].value_counts()[ad_id], ad_id))
    print('[*] # valid users = {}'.format(len(unique(df_log['user']))))

    for start_round in start_rounds:
        assert 1 <= start_round <= num_allowed_rounds
        print('[*] processing dataset for attack started at {}'.format(
            start_time + datetime.timedelta(days=start_round)
        ))

        for time_window in time_windows:
            # generate (i) victim's dynamic set per day for each time_window
            #          (ii) the positive/negative records w.r.t. start_round
            print('    - set time window = {} day(s)'.format(time_window))
            victim_set_lists[time_window] = list()

            for round_counter in range(1, num_allowed_rounds + 1):
                prev_timestamp = max(
                    start_time.timestamp(),
                    (start_time + datetime.timedelta(days=round_counter - time_window)).timestamp()
                )
                curr_timestamp = (start_time + datetime.timedelta(days=round_counter)).timestamp()

                # collect users within the time window
                users = set(np.array(
                    df_log.loc[
                        (df_log['ad_id'] == ad_id) &
                        (prev_timestamp <= df_log['timestamp']) & (df_log['timestamp'] <= curr_timestamp),
                        ['user']
                    ]['user']
                ))
                print('        > from {} to {}: {} distinct ad viewers'.format(
                    datetime.datetime.fromtimestamp(prev_timestamp),
                    datetime.datetime.fromtimestamp(curr_timestamp),
                    len(users)
                ))
                victim_set_lists[time_window].append(users)

            # set betas
            ratio = len(victim_set_lists[time_window][start_round - 1]) / len(unique(df_log['user']))
            betas[start_round] = [
                round(ratio, 4), round(5 * ratio, 4), round(10 * ratio, 4)
            ]
            print('    - start round = {}, betas = {}'.format(start_round, betas[start_round]))

            # compute mutual information
            df_featured_population = copy.deepcopy(df_featured_user)
            calculate_mutual_information(
                df_featured_population=df_featured_population, features=features,
                victim_set=victim_set_lists[time_window][start_round - 1]
            )

            for beta in betas[start_round]:
                num_sampled_viewers = ceil(beta * attacker_set_size)
                num_sampled_non_viewers = attacker_set_size - num_sampled_viewers
                print('        > beta = {}, attacker\'s set size = {}'.format(beta, attacker_set_size))
                print('        > uniformly draw {} viewers and {} non-viewers'.format(
                    num_sampled_viewers, num_sampled_non_viewers
                ))

                attacker_set_lists[(beta, start_round, time_window)] = list()
                for iteration_counter in range(num_iterations):
                    # generate a sample attacker's set for each iteration
                    sampled_viewers = df_featured_user.loc[
                        df_featured_user['user'].isin(victim_set_lists[time_window][start_round - 1])
                    ].sample(n=num_sampled_viewers, replace=False)
                    sampled_non_viewers = df_featured_user.loc[
                        ~df_featured_user['user'].isin(victim_set_lists[time_window][start_round - 1])
                    ].sample(n=num_sampled_non_viewers, replace=False)

                    sampled_attacker_set = shuffle(sampled_viewers.append(sampled_non_viewers))
                    assert sampled_attacker_set.shape[0] == sampled_viewers.shape[0] + sampled_non_viewers.shape[0]
                    assert sampled_attacker_set.isnull().sum().sum() == 0

                    attacker_set_lists[(beta, start_round, time_window)].append(sampled_attacker_set)

                print()

    return period, attacker_set_lists, victim_set_lists, set(unique(df_log['user']))
