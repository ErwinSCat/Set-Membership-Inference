import numpy as np
import traceback

from numpy import unique, where
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from utils import PriorityQueue, from_log_points_to_time_series


max_num_allowed_rounds = 1000


def reveal_intersection_size(node, victim_sets, num_allowed_rounds, round_counter, system_run_counter,
                             num_queries_per_victim_set, is_dynamic):
    # # system runs that have been consumed
    if system_run_counter < num_allowed_rounds * num_queries_per_victim_set:
        assert node.data['my_intersection_size'] is None
        node.data['my_intersection_size'] = len(
            node.data['stored_set'].intersection(victim_sets[round_counter if is_dynamic else 0])
        )

        flag = ((system_run_counter + 1) % num_queries_per_victim_set == 0)
        new_round_counter = (round_counter + 1) if flag else round_counter
        new_system_run_counter = system_run_counter + 1
        return new_round_counter, new_system_run_counter
    else:
        raise Exception('SYSTEM_RUN_CONSUMED')


def random_partition(raw_data, num_clusters=2):
    shuffled = shuffle(raw_data)
    return np.array_split(shuffled, min(num_clusters, shuffled.shape[0]))


def kmeans_partition(raw_data, num_clusters=2):
    input_data_copy = raw_data.drop(['user'], axis=1)

    # standardization
    scaler = StandardScaler()
    input_data_copy_array = scaler.fit_transform(input_data_copy)

    # if all items in input_data_copy_array are identical, perform random partition
    input_data_copy_array_unique = unique(input_data_copy_array, axis=0)
    if input_data_copy_array_unique.shape[0] < num_clusters:
        return random_partition(raw_data=raw_data, num_clusters=num_clusters)
    else:  # run K-means
        ret = list()
        sum_check = 0
        model = KMeans(n_clusters=num_clusters)
        model.fit(input_data_copy_array_unique)
        y = model.predict(input_data_copy_array)
        clusters = unique(y)
        for cluster in clusters:
            row_x = where(y == cluster)
            ret.append(raw_data.iloc[row_x])
            sum_check += ret[-1].shape[0]
            assert ret[-1].shape[0] > 0

        assert len(ret) == len(clusters) and sum_check == raw_data.shape[0]
        return ret


def DBSCAN_partition(raw_data):
    input_data_copy = raw_data.drop(['user'], axis=1)

    # standardization
    scaler = StandardScaler()
    input_data_copy_array = scaler.fit_transform(input_data_copy)

    # if all items in input_data_copy_array are identical, perform random partition
    input_data_copy_array_unique = unique(input_data_copy_array, axis=0)
    if input_data_copy_array_unique.shape[0] < 2:
        return random_partition(raw_data=raw_data, num_clusters=2)
    else:  # run DBSCAN
        ret = list()
        sum_check = 0
        model = DBSCAN(eps=0.30, min_samples=9)
        y = model.fit_predict(input_data_copy_array)
        clusters = unique(y)
        if len(clusters) == 1:
            return random_partition(raw_data=raw_data, num_clusters=2)
        else:
            for cluster in clusters:
                row_x = where(y == cluster)
                ret.append(raw_data.iloc[row_x])
                sum_check += ret[-1].shape[0]
                assert ret[-1].shape[0] > 0

            assert len(ret) == len(clusters) and sum_check == raw_data.shape[0]
            return ret


# start_round: counted from 1
def blueprint(tree, victim_sets, start_round, num_queries_per_victim_set, is_dynamic=True):
    # parameter check
    assert 1 <= start_round <= len(victim_sets)
    assert isinstance(victim_sets, list)
    for item in victim_sets:
        assert isinstance(item, set)

    # init
    round_counter = start_round - 1
    system_run_counter = 0
    members = set()
    non_members = set()
    unknowns = tree.root.data['stored_set']

    member_log_points = list()
    non_member_log_points = list()

    num_allowed_rounds = (len(victim_sets) - start_round + 1) if is_dynamic else max_num_allowed_rounds
    # print('    - allowed number of system runs = {}'.format(num_allowed_rounds))

    try:
        round_counter, system_run_counter = reveal_intersection_size(
            node=tree.root, victim_sets=victim_sets, num_allowed_rounds=num_allowed_rounds,
            round_counter=round_counter, system_run_counter=system_run_counter,
            num_queries_per_victim_set=num_queries_per_victim_set,
            is_dynamic=is_dynamic
        )

        forest = PriorityQueue()
        forest.push(priority=priority_score(tree.root), item=tree.root)

        while forest.is_empty() is False:
            node = forest.pop()
            while 0 < node.data['my_intersection_size'] < len(node.data['stored_set']):
                child_nodes = tree.tree.children(node.identifier)
                child_nodes.sort(key=lambda x: len(x.data['stored_set']))

                summed_intersection_sizes = 0
                for i in range(len(child_nodes) - 1):
                    round_counter, system_run_counter = reveal_intersection_size(
                        node=child_nodes[i], victim_sets=victim_sets, num_allowed_rounds=num_allowed_rounds,
                        round_counter=round_counter, system_run_counter=system_run_counter,
                        num_queries_per_victim_set=num_queries_per_victim_set,
                        is_dynamic=is_dynamic
                    )
                    summed_intersection_sizes += child_nodes[i].data['my_intersection_size']

                # locally set the last child node's intersection size
                child_nodes[len(child_nodes) - 1].data['my_intersection_size'] \
                    = node.data['my_intersection_size'] - summed_intersection_sizes

                # find the child node with the greatest priority
                sort_queue = PriorityQueue()
                for child_node in child_nodes:
                    sort_queue.push(priority=priority_score(child_node), item=child_node)

                # set node to be the child node with the greatest priority
                # & push each other node into the priority queue / set all its elements as non-members
                node = sort_queue.pop()
                while sort_queue.is_empty() is False:
                    other_node = sort_queue.pop()
                    # forest.push(priority=priority_score(other_node), item=other_node)
                    if priority_score(other_node) <= 0:
                        non_members = non_members.union(other_node.data['stored_set'])
                        non_member_log_points.append((system_run_counter, len(non_members)))
                    else:
                        forest.push(priority=priority_score(other_node), item=other_node)

            if node.data['my_intersection_size'] > 0:
                members = members.union(node.data['stored_set'])
                member_log_points.append((system_run_counter, len(members)))
            # else:
            #     non_members = non_members.union(node.data['stored_set'])
            #     non_member_log_points.append((system_run_counter, len(non_members)))

    except Exception as err:
        if str(err) != 'SYSTEM_RUN_CONSUMED':
            traceback.print_exc()
    finally:
        unknowns = unknowns.difference(members)
        unknowns = unknowns.difference(non_members)

        # from log points to time series
        member_time_series = [0] * ((start_round - 1) * num_queries_per_victim_set)
        member_time_series.extend(from_log_points_to_time_series(member_log_points))
        non_member_time_series = [0] * ((start_round - 1) * num_queries_per_victim_set)
        non_member_time_series.extend(from_log_points_to_time_series(non_member_log_points))
        assert len(member_time_series) == 0 or member_time_series[-1] == len(members)
        assert len(non_member_time_series) == 0 or non_member_time_series[-1] == len(non_members)

        while len(member_time_series) < ((len(victim_sets) * num_queries_per_victim_set) if is_dynamic else max_num_allowed_rounds):
            member_time_series.append(len(members))
        while len(non_member_time_series) < ((len(victim_sets) * num_queries_per_victim_set) if is_dynamic else max_num_allowed_rounds):
            non_member_time_series.append(len(non_members))

        print('    - attack terminates after {} system runs'.format(system_run_counter))
        return system_run_counter, members, non_members, unknowns, member_time_series, non_member_time_series


def get_attacker_type(partition_method):
    attacker_type = {
        random_partition: 'baseline',
        kmeans_partition: 'k-means',
        DBSCAN_partition: 'DBSCAN',
        toy_attack: 'toy'
    }
    return attacker_type[partition_method]


def priority_score(node):
    return node.data['my_intersection_size'] / len(node.data['stored_set'])


def toy_attack(attacker_set, victim_sets, start_round, num_queries_per_victim_set, is_dynamic=True):
    assert 1 <= start_round <= len(victim_sets)
    assert isinstance(attacker_set, set)
    assert isinstance(victim_sets, list)
    for item in victim_sets:
        assert isinstance(item, set)

    element_list = list(attacker_set)
    element_list = shuffle(element_list)

    # init
    system_run_counter = 0
    members = set()
    non_members = set()
    unknowns = attacker_set

    member_time_series = [0] * ((start_round - 1) * num_queries_per_victim_set)
    non_member_time_series = [0] * ((start_round - 1) * num_queries_per_victim_set)

    counter_list = list()
    for round_counter in range(start_round - 1, len(victim_sets) if is_dynamic else max_num_allowed_rounds):
        for _ in range(num_queries_per_victim_set):
            counter_list.append(round_counter)

    for round_counter in counter_list:
        if system_run_counter < len(element_list):
            if element_list[system_run_counter] in victim_sets[round_counter if is_dynamic else 0]:
                members.add(element_list[system_run_counter])
            else:
                non_members.add(element_list[system_run_counter])

            system_run_counter += 1

        member_time_series.append(len(members))
        non_member_time_series.append(len(non_members))

    unknowns = unknowns.difference(members)
    unknowns = unknowns.difference(non_members)

    print('    - attack terminates after {} system runs'.format(system_run_counter))
    return system_run_counter, members, non_members, unknowns, member_time_series, non_member_time_series
