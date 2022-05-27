from queue import Queue
import numpy as np
import pandas as pd
from treelib import Tree, Node


def _get_curr_id(args=None):
    PartitionTree.id_counter += 1
    return 'node-' + str(PartitionTree.id_counter - 1)


class PartitionTree:
    id_counter = 0

    def __init__(self, partition_func, root_data, is_print=False):
        assert isinstance(root_data, pd.DataFrame)
        self.partition_func = partition_func
        self.tree = Tree()
        self.root = Node(
            data={
                'raw_data': root_data,
                'stored_set': set(np.array(root_data['user'])),
                'subtree_height': None,
                'my_intersection_size': None
            },
            identifier=_get_curr_id()
        )
        self.tree.add_node(self.root)
        self.partition()
        self.update_subtree_height(self.root)
        if is_print:
            self.tree.show()

    def partition(self):
        q = Queue()
        q.put(self.root)
        while q.empty() is False:
            node = q.get()
            partitioned_data_list = self.partition_func(node.data['raw_data'])
            for i in range(len(partitioned_data_list)):
                child_node = Node(
                    data={
                        'raw_data': partitioned_data_list[i],
                        'stored_set': set(np.array(partitioned_data_list[i]['user'])),
                        'subtree_height': None,
                        'my_intersection_size': None
                    },
                    identifier=_get_curr_id()
                )
                self.tree.add_node(child_node, node)

                if partitioned_data_list[i].shape[0] > 1:
                    q.put(child_node)

    def update_subtree_height(self, curr_node):
        if curr_node.is_leaf() is True:
            curr_node.data['subtree_height'] = 0
            curr_node.tag = curr_node.identifier + ', height = ' + str(curr_node.data['subtree_height'])
            return 0
        else:
            child_nodes = self.tree.children(curr_node.identifier)
            child_node_heights = list()
            for child_node in child_nodes:
                child_node_heights.append(self.update_subtree_height(child_node))

            curr_node.data['subtree_height'] = max(child_node_heights) + 1
            curr_node.tag = curr_node.identifier + ', height = ' + str(curr_node.data['subtree_height'])
            return curr_node.data['subtree_height']
