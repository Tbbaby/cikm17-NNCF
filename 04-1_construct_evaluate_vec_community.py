import pandas as pd
import numpy as np
import networkx as nx
from collections import Counter
import networkx as nx
import community
import sys

"""
generate the test evaluation data 
Output file:
uninteract_vec_element

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


if __name__=="__main__":
    print '=================04-1: Generating test data ==================='
    #N = 99
    #Max_Num_Neighbor =50
    #resolution = 0.5
    N =int(sys.argv[1])
    Max_Num_Neighbor =int(sys.argv[2])
    resolution = float(sys.argv[3])
    print 'N_test_negative: ', N
    print 'max_neighbors: ', Max_Num_Neighbor
    print 'resolution: ', resolution
    relation_train_positive = pd.read_csv('./Data/relation_train_positive.csv')
    relation_test = pd.read_csv('./Data/relation_test.csv')
    all_item = pd.read_csv('./Data/all_item.csv')
    all_item = all_item.item.values
    G_all = nx.Graph()
    G_all.add_edges_from(relation_train_positive.values)
    G_all.add_edges_from(relation_test.values)

    test_user = relation_test.user.values

    uninteract_user_list = []
    uninteract_item_list = []

    for user in test_user:
        interact_item = nx.neighbors(G_all,user)
        un_interact_item = list(set(all_item) - set(interact_item))
        np.random.shuffle(un_interact_item)
        if len(un_interact_item) < N:
            un_interact_item = un_interact_item * (int(N / len(un_interact_item))+1)
        un_interact_item_choose = un_interact_item[:N]
        uninteract_user_list += [user] * N
        uninteract_item_list += un_interact_item_choose

    uninteract = pd.DataFrame(np.array([uninteract_user_list,uninteract_item_list]).T , columns = ['user','item'])

    G = nx.Graph()
    G.add_edges_from(relation_train_positive.values)

    partition = community.best_partition(G,resolution=resolution)

    community_dict ={}
    community_dict.setdefault(0,[])
    for i in range(len(partition.values())):
        community_dict[i] = []
    for node,part in partition.items():
        community_dict[part] = community_dict[part] + [node]

    prepare_vector_element_test(G,relation_test, 'relation_test_vec_element')
    prepare_vector_element_test(G,uninteract, 'uninteract_vec_element')
    print 'test samples:', len(relation_test)
    print 'test uninteract samples:', len(uninteract)
    print 'Generate test data successfully!'
