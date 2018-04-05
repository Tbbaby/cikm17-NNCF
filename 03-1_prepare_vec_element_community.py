import pandas as pd
import numpy as np
import networkx as nx
from collections import Counter
import networkx as nx
import community
import os
import sys



"""
To construct community neighbors of NNCF
Output files: 
relation_test_vec_element.csv
relation_train_vec_element_all_data.csv
"""

def get_community_member(partition,community_dict,node,kind):
    comm = community_dict[partition[node]]
    return [x for x in comm if x.startswith(kind)]


def prepare_vector_element_test(G, relation, filename):
    f = open('./Data/' + filename + '.txt', 'wb')
    for r in relation.values:
        line = ''
        user, item = r
        line = line + user.split('_')[1] + ','
        line = line + item.split('_')[1] + '|'
        try:
            item2user_neighbor = get_community_member(partition, community_dict, user, 'u')
            np.random.shuffle(item2user_neighbor)
            if len(item2user_neighbor) > Max_Num_Neighbor:
                item2user_neighbor = item2user_neighbor[:Max_Num_Neighbor]
        except:
            item2user_neighbor = []

        item2user_neighbor = ','.join([str(x.split('_')[1]) for x in item2user_neighbor])
        line = line + item2user_neighbor + '|'

        try:
            user2item_neighbor = get_community_member(partition, community_dict, item, 'i')
            np.random.shuffle(user2item_neighbor)
            if len(user2item_neighbor) > Max_Num_Neighbor:
                user2item_neighbor = user2item_neighbor[:Max_Num_Neighbor]
        except:
            user2item_neighbor = []

        user2item_neighbor = ','.join([str(x.split('_')[1]) for x in user2item_neighbor])
        line = line + user2item_neighbor

        f.writelines(line)
        f.writelines('\n')
    f.close()


def prepare_vector_element_train(G, relation, label, filename):
    f = open('./Data/' + filename + '.txt', 'wb')
    for r, l in zip(relation.values, label.values):
        line = str(l[0]) + '|'
        user, item = r
        line = line + user.split('_')[1] + ','
        line = line + item.split('_')[1] + '|'

        try:
            item2user_neighbor = get_community_member(partition, community_dict, user, 'u')
            np.random.shuffle(item2user_neighbor)
            if len(item2user_neighbor) > Max_Num_Neighbor:
                item2user_neighbor = item2user_neighbor[:Max_Num_Neighbor]
        except:
            item2user_neighbor = []

        item2user_neighbor = ','.join([str(x.split('_')[1]) for x in item2user_neighbor])
        line = line + item2user_neighbor + '|'

        try:
            user2item_neighbor = get_community_member(partition, community_dict, item, 'i')
            np.random.shuffle(user2item_neighbor)
            if len(user2item_neighbor) > Max_Num_Neighbor:
                user2item_neighbor = user2item_neighbor[:Max_Num_Neighbor]
        except:
            user2item_neighbor = []

        user2item_neighbor = ','.join([str(x.split('_')[1]) for x in user2item_neighbor])
        line = line + user2item_neighbor

        f.writelines(line)
        f.writelines('\n')
    f.close()


if __name__=="__main__":
    print '=========03-1: Construct Community neighbors of NNCF========='
    Max_Num_Neighbor = int(sys.argv[1])
    resolution=float(sys.argv[2])
    print 'max_neighbors: ', Max_Num_Neighbor
    print 'resolution: ', resolution
    #Max_Num_Neighbor = 50
    # the community partition parameter
    #resolution = 0.5
    relation_train_positive = pd.read_csv('./Data/relation_train_positive.csv')
    relation_train = pd.read_csv('./Data/relation_train.csv')
    relation_test = pd.read_csv('./Data/relation_test.csv')
    relation_train_label = pd.read_csv('./Data/relation_train_label.csv')

    G = nx.Graph()
    G.add_edges_from(relation_train_positive.values)
    partition = community.best_partition(G,resolution=resolution)
    print 'community num : %d'%(len(set(partition.values())))
    X = raw_input('continue?  [y/n] ')

    if X == 'n':
        os._exit(0)
    else:
        pass

    community_dict ={}
    community_dict.setdefault(0,[])
    for i in range(len(partition.values())):
        community_dict[i] = []
    for node,part in partition.items():
        community_dict[part] = community_dict[part] + [node]

    prepare_vector_element_test(G, relation_test, 'relation_test_vec_element')
    train_original_len = len(relation_train)
    prepare_vector_element_train(G, relation_train, relation_train_label, 'relation_train_vec_element_all_data')
    print 'Community neighbors of NNCF are generated successfully!'
