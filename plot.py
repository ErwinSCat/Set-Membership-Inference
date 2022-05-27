import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import os
import copy
from collections import OrderedDict

from attack import get_attacker_type


linestyle = {
    'baseline': '--',
    'k-means': '-',
    'DBSCAN': '-.',
    'toy': ':'
}


def plot_from_data(dataset_params):
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

    for time_window in time_windows:  # for each time window
        for start_round in start_rounds:  # for each start round
            plt.figure()
            fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(16, 3.4))

            for beta in betas[start_round]:  # for each beta
                for partition_method in partition_methods:  # for each attacker type
                    plot_data_dir = './plot_data/{}/{}/'.format(dataset_name, get_attacker_type(partition_method=partition_method))

                    # read data
                    df_total = pd.read_csv(plot_data_dir + file_name.format(
                        attacker_set_size, beta, time_window, start_round, num_queries_per_victim_set
                    ) + ', total.csv')
                    df_member = pd.read_csv(plot_data_dir + file_name.format(
                        attacker_set_size, beta, time_window, start_round, num_queries_per_victim_set
                    ) + ', member.csv')
                    # df_non_member = pd.read_csv(plot_data_dir + file_name.format(
                    #     attacker_set_size, beta, time_window, start_round, num_queries_per_victim_set
                    # ) + ', non_member.csv')

                    assert df_total.shape[0] == num_iterations
                    assert df_member.shape[0] == num_iterations
                    # assert df_non_member.shape[0] == num_iterations

                    # generate plot data
                    period = list(df_total.keys())
                    x = np.arange(1, len(period) + 1)
                    mean_data_total = np.array(df_total.mean(axis=0))
                    std_data_total = np.array(df_total.std(axis=0))
                    mean_data_num_members = np.array(df_member.mean(axis=0))
                    std_data_num_members = np.array(df_member.std(axis=0))
                    # mean_data_num_non_members = np.array(df_non_member.mean(axis=0))
                    # std_data_num_non_members = np.array(df_non_member.std(axis=0))

                    # plot
                    axs[betas[start_round].index(beta)].plot(
                        x, mean_data_total, color='dimgray',
                        label='# total ({})'.format(get_attacker_type(partition_method=partition_method)),
                        linestyle=linestyle[get_attacker_type(partition_method=partition_method)]
                    )
                    # axs[betas[start_round].index(beta)].fill_between(
                    #     x,
                    #     mean_data_total - std_data_total,
                    #     mean_data_total + std_data_total,
                    #     color='dimgray', alpha=0.1
                    # )
                    axs[betas[start_round].index(beta)].plot(
                        x, mean_data_num_members, color='red',
                        label='# members ({})'.format(get_attacker_type(partition_method=partition_method)),
                        linestyle=linestyle[get_attacker_type(partition_method=partition_method)]
                    )
                    # axs[betas[start_round].index(beta)].fill_between(
                    #     x,
                    #     mean_data_num_members - std_data_num_members,
                    #     mean_data_num_members + std_data_num_members,
                    #     color='red', alpha=0.1
                    # )
                    # axs[betas[start_round].index(beta)].plot(
                    #     x, mean_data_num_non_members, color='blue',
                    #     label='# non-members ({})'.format(get_attacker_type(partition_method=partition_method)),
                    #     linestyle=linestyle[get_attacker_type(partition_method=partition_method)]
                    # )
                    # axs[betas[start_round].index(beta)].fill_between(
                    #     x,
                    #     mean_data_num_non_members - std_data_num_non_members,
                    #     mean_data_num_non_members + std_data_num_non_members,
                    #     color='blue', alpha=0.1
                    # )

                    axs[betas[start_round].index(beta)].grid(b='on')
                    axs[betas[start_round].index(beta)].set_title(
                        '|X| = {}, $\\beta$ = {}'.format(attacker_set_size, beta)
                    )

                    # TODO: set x ticks & labels
                    sep = None
                    if dataset_name == 'covid_contact_tracing':
                        sep = num_queries_per_victim_set * 3
                        period = [period[i].replace('2020-', '') for i in range(len(period))]
                        period.append('05-01')
                    elif dataset_name == 'ad_conversion_revenue':
                        sep = num_queries_per_victim_set * 24 * 1
                        period = [period[i].replace('2017-', '') for i in range(len(period))]
                    elif dataset_name == 'ad_conversion_lift':
                        sep = num_queries_per_victim_set * 3
                        period = [period[i].replace('2019-', '') for i in range(len(period))]

                    xticks = [1 + i for i in range((start_round - 1) * num_queries_per_victim_set, len(period) + 1, sep)]
                    xlabels = [period[i - 1] for i in xticks]
                    axs[betas[start_round].index(beta)].set_xticks(xticks)
                    axs[betas[start_round].index(beta)].set_xticklabels(xlabels, rotation=30, ha='right')
                    axs[betas[start_round].index(beta)].set_xlim((start_round - 1) * num_queries_per_victim_set + 1, len(period))
                    axs[betas[start_round].index(beta)].set_ylim(1, 1.5 * attacker_set_size)
                    axs[betas[start_round].index(beta)].set_yscale('log')

            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            fig.legend(by_label.values(), by_label.keys(),
                       loc='center', bbox_to_anchor=(0.5, 1.01), ncol=2 * len(partition_methods))

            plot_dir = './plot/{}/'.format(dataset_name)
            os.makedirs(plot_dir, exist_ok=True)
            plt.subplots_adjust(wspace=0.27)
            plt.savefig(
                plot_dir + 'X size = {}, time window = {}, start round = {}, features = {}, num. queries = {}.pdf'.format(
                    attacker_set_size, time_window, start_round, features, num_queries_per_victim_set
                ), bbox_inches='tight'
            )
            plt.close()


def plot_snapshot_set_size(dataset_name, time_window, num_queries_per_victim_set, period, victim_set_lists):
    plt.figure(figsize=(6, 2))
    y = np.array([len(victim_set) for victim_set in victim_set_lists[time_window]])

    period_copy = copy.deepcopy(period)

    dates = None
    ylabel = None
    if dataset_name == 'covid_contact_tracing':
        ylabel = '# infected persons'
        index = [i for i in range(0, len(period_copy), num_queries_per_victim_set)]
        dates = [np.datetime64(date.replace(' 1/{}'.format(num_queries_per_victim_set), '')) for date in [period_copy[i] for i in index]]
        plt.ylim(0, 8000)

    elif dataset_name == 'ad_conversion_revenue':
        ylabel = '# distinct clickers'
        dates = [np.datetime64(date) for date in period_copy]
        plt.ylim(0, 8000)

    elif dataset_name == 'ad_conversion_lift':
        ylabel = '# distinct viewers'
        dates = [np.datetime64(date) for date in period_copy]
        plt.ylim(0, 60000)

    plt.plot(dates, y)
    plt.grid()
    plt.ylabel(ylabel)

    ax = plt.gca()
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())
    )

    plot_dir = './plot/{}/'.format(dataset_name)
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(
        plot_dir + 'time window = {}, num. queries = {}.pdf'.format(
            time_window, num_queries_per_victim_set
        ), bbox_inches='tight'
    )
    plt.close()
