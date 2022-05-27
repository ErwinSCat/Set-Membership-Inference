from load_data import *
from tree import PartitionTree
from attack import blueprint, get_attacker_type
from plot import plot_snapshot_set_size

import os
from math import floor


def process_dynamic_dataset(dataset_params):
    dataset_load_func = {
        'covid_contact_tracing': load_Israel_covid_dataset,
        'ad_conversion_revenue': load_taobao_dataset,
        'ad_conversion_lift': load_tencent_dataset
    }

    dataset_population_size = {
        'covid_contact_tracing': 255668,
        'ad_conversion_revenue': 1061768,
        'ad_conversion_lift': 1341958
    }

    # parse parameters
    dataset_name = dataset_params['dataset_name']
    file_name = dataset_params['file_name']

    attacker_set_size = dataset_params['attacker_set_size']
    num_iterations = dataset_params['num_iterations']
    partition_methods = dataset_params['partition_methods']
    betas = dataset_params['betas']
    features = dataset_params['features']
    time_windows = dataset_params['time_windows']
    start_rounds = dataset_params['start_rounds']
    num_queries_per_victim_set = dataset_params['num_queries_per_victim_set']

    # load dataset
    period, attacker_set_lists, victim_set_lists, population = dataset_load_func[dataset_name](
        attacker_set_size=attacker_set_size,
        num_iterations=num_iterations,
        betas=betas,
        features=features,
        time_windows=time_windows,
        start_rounds=start_rounds,

        others=dataset_params
    )
    assert len(population) == dataset_population_size[dataset_name]

    # run attack for each parameter setting
    for time_window in time_windows:  # for each time_window
        # plot size of victim's snapshot sets
        plot_snapshot_set_size(
            dataset_name=dataset_name, time_window=time_window, num_queries_per_victim_set=num_queries_per_victim_set,
            period=period, victim_set_lists=victim_set_lists
        )

        assert len(period) == len(victim_set_lists[time_window]) * num_queries_per_victim_set

        for start_round in start_rounds:  # for each start_round
            for beta in betas[start_round]:  # for each beta
                for partition_method in partition_methods:  # for each attacker type
                    # plot data
                    total_time_series_list = list()
                    member_time_series_list = list()
                    non_member_time_series_list = list()
                    # statistics
                    final_num_members = list()          # # all members retrieved
                    final_num_non_members = list()      # # all non-members retrieved
                    final_num_true_positives = list()   # # TP
                    final_num_false_positives = list()  # # FP
                    final_num_true_negatives = list()   # # TN
                    final_num_false_negatives = list()  # # FN

                    tree_height = list()
                    root_intersection_size = list()

                    ratio_true_positive_from_begin = list()
                    ratio_true_positive_recovered = list()

                    # for each iteration
                    for iteration_counter in range(num_iterations):
                        print('[*] {} attack #{}, X size = {}, beta = {}, time window = {}, start round = {}'.format(
                            get_attacker_type(partition_method=partition_method), iteration_counter + 1,
                            attacker_set_size, beta, time_window, start_round
                        ))

                        # launch the attack
                        if get_attacker_type(partition_method=partition_method) != 'toy':
                            tree = PartitionTree(
                                partition_method,
                                attacker_set_lists[(beta, start_round, time_window)][iteration_counter]
                            )

                            system_run_counter, members, non_members, unknowns, member_time_series, non_member_time_series = blueprint(
                                tree=tree,
                                victim_sets=victim_set_lists[time_window],
                                start_round=start_round,
                                num_queries_per_victim_set=num_queries_per_victim_set
                            )

                            print('    - tree height = {}'.format(tree.root.data['subtree_height']))
                            tree_height.append(tree.root.data['subtree_height'])
                            print('    - root intersection size = {}'.format(tree.root.data['my_intersection_size']))
                            root_intersection_size.append(tree.root.data['my_intersection_size'])

                        else:  # toy attack
                            system_run_counter, members, non_members, unknowns, member_time_series, non_member_time_series = partition_method(
                                attacker_set=set(np.array(
                                    attacker_set_lists[(beta, start_round, time_window)][iteration_counter]['user']
                                )),
                                victim_sets=victim_set_lists[time_window],
                                start_round=start_round,
                                num_queries_per_victim_set=num_queries_per_victim_set
                            )

                        # collect positives and negatives
                        positives = set()
                        negatives = population.copy()
                        round_range = list(range(start_round, start_round + floor((system_run_counter - 1) / num_queries_per_victim_set) + 1))
                        print('    - rounds: {}'.format(round_range))
                        for round_counter in range(1, len(victim_set_lists[time_window]) + 1):
                            if round_counter in round_range:
                                positives = positives.union(victim_set_lists[time_window][round_counter - 1])

                        negatives = negatives.difference(positives)
                        assert len(positives) + len(negatives) == len(population)
                        print('    - # positives = {}, # negatives = {}'.format(len(positives), len(negatives)))

                        # update plot data & statistics
                        total_time_series_list.append(np.array(member_time_series) + np.array(non_member_time_series))
                        member_time_series_list.append(np.array(member_time_series))
                        non_member_time_series_list.append(np.array(non_member_time_series))

                        final_num_members.append(len(members))
                        final_num_non_members.append(len(non_members))

                        final_num_true_positives.append(len(members.intersection(positives)))
                        final_num_false_positives.append(len(members.intersection(negatives)))
                        final_num_true_negatives.append(len(non_members.intersection(negatives)))
                        final_num_false_negatives.append(len(non_members.intersection(positives)))

                        ratio_true_positive_from_begin.append(
                            (len(victim_set_lists[time_window][start_round - 1].intersection(members.intersection(positives)))
                             / len(members.intersection(positives))) if len(members.intersection(positives)) != 0 else np.nan
                        )
                        ratio_true_positive_recovered.append(
                            len(victim_set_lists[time_window][start_round - 1].intersection(members.intersection(positives)))
                            / ceil(attacker_set_size * beta)
                        )
                        assert len(victim_set_lists[time_window][start_round - 1].intersection(set(np.array(
                                    attacker_set_lists[(beta, start_round, time_window)][iteration_counter]['user']
                                )))) == ceil(attacker_set_size * beta)

                        assert len(members) == len(members.intersection(positives)) + len(members.intersection(negatives))
                        assert len(non_members) == len(non_members.intersection(negatives)) + len(non_members.intersection(positives))
                        assert len(members) + len(non_members) + len(unknowns) == attacker_set_size

                    # output statistics
                    print('[*] final statistics')
                    print('    - {} attack: X size = {}, beta = {}, time window = {}, start round = {}'.format(
                        get_attacker_type(partition_method=partition_method),
                        attacker_set_size, beta, time_window, start_round
                    ))
                    if get_attacker_type(partition_method=partition_method) != 'toy':
                        print('    - tree height, mean (std) = {:.1f} ({:.1f})'.format(
                            (np.array(tree_height)).mean(), (np.array(tree_height)).std()
                        ))
                        print('    - root intersection size, mean (std) = {:.1f} ({:.1f})'.format(
                            (np.array(root_intersection_size)).mean(), (np.array(root_intersection_size)).std()
                        ))

                    print('    - # TP from initial intersection / # TP, mean (std) = {:.2f} ({:.2f})'.format(
                        (np.array(ratio_true_positive_from_begin)).mean(), (np.array(ratio_true_positive_from_begin)).std()
                    ))
                    print('    - # TP from initial intersection / initial intersection size, mean (std) = {:.2f} ({:.2f})'.format(
                        (np.array(ratio_true_positive_recovered)).mean(), (np.array(ratio_true_positive_recovered)).std()
                    ))

                    print('    - # members, mean (std) = {:.1f} ({:.1f})'.format(
                        (np.array(final_num_members)).mean(), (np.array(final_num_members)).std()
                    ))
                    print('    - # non-members, mean (std) = {:.1f} ({:.1f})'.format(
                        (np.array(final_num_non_members)).mean(), (np.array(final_num_non_members)).std()
                    ))
                    print('    - # true positives, mean (std) = {:.1f} ({:.1f})'.format(
                        (np.array(final_num_true_positives)).mean(), (np.array(final_num_true_positives)).std()
                    ))
                    print('    - # false positives, mean (std) = {:.1f} ({:.1f})'.format(
                        (np.array(final_num_false_positives)).mean(), (np.array(final_num_false_positives)).std()
                    ))
                    print('    - # true negatives, mean (std) = {:.1f} ({:.1f})'.format(
                        (np.array(final_num_true_negatives)).mean(), (np.array(final_num_true_negatives)).std()
                    ))
                    print('    - # false negatives, mean (std) = {:.1f} ({:.1f})'.format(
                        (np.array(final_num_false_negatives)).mean(), (np.array(final_num_false_negatives)).std()
                    ))

                    print()

                    # output plot data
                    plot_data_dir = './plot_data/{}/{}/'.format(dataset_name, get_attacker_type(partition_method=partition_method))
                    os.makedirs(plot_data_dir, exist_ok=True)
                    df_total = pd.DataFrame(total_time_series_list)
                    df_member = pd.DataFrame(member_time_series_list)
                    df_non_member = pd.DataFrame(non_member_time_series_list)

                    df_total.columns = period
                    df_member.columns = period
                    df_non_member.columns = period

                    df_total.to_csv(plot_data_dir + file_name.format(
                        attacker_set_size, beta, time_window, start_round, num_queries_per_victim_set
                    ) + ', total.csv', index=False)
                    df_member.to_csv(plot_data_dir + file_name.format(
                        attacker_set_size, beta, time_window, start_round, num_queries_per_victim_set
                    ) + ', member.csv', index=False)
                    df_non_member.to_csv(plot_data_dir + file_name.format(
                        attacker_set_size, beta, time_window, start_round, num_queries_per_victim_set
                    ) + ', non_member.csv', index=False)
