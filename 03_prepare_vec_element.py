import pandas as pd
import numpy as np
import networkx as nx
from collections import Counter
import sys


"""
To construct direct neighbors of NNCF
Output files: 
relation_test_vec_element.csv
relation_train_vec_element_all_data.csv
"""


def prepare_vector_element_test(G, relation, filename):
    f = open('./Data/' + filename + '.txt', 'wb')
    for r in relation.values:
        line = ''
        user, item = r
        line = line + user.split('_')[1] + ','
        line = line + item.split('_')[1] + '|'
        try:
            user2item_neighbor = nx.neighbors(G, item)
            np.random.shuffle(user2item_neighbor)

            if len(user2item_neighbor) > Max_Num_Neighbor:
                user2item_neighbor = user2item_neighbor[:Max_Num_Neighbor]

        except:
            user2item_neighbor = []

        user2item_neighbor = ','.join([str(x.split('_')[1]) for x in user2item_neighbor])
        line = line + user2item_neighbor + '|'

        try:
            item2user_neighbor = nx.neighbors(G, user)
            np.random.shuffle(item2user_neighbor)

            if len(item2user_neighbor) > Max_Num_Neighbor:
                item2user_neighbor = item2user_neighbor[:Max_Num_Neighbor]

        except:
            item2user_neighbor = []

        item2user_neighbor = ','.join([str(x.split('_')[1]) for x in item2user_neighbor])
        line = line + item2user_neighbor

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
            user2item_neighbor = nx.neighbors(G, item)
            np.random.shuffle(user2item_neighbor)

            if len(user2item_neighbor) > Max_Num_Neighbor:
                user2item_neighbor = user2item_neighbor[:Max_Num_Neighbor]

        except:
            user2item_neighbor = []

        user2item_neighbor = ','.join([str(x.split('_')[1]) for x in user2item_neighbor])
        line = line + user2item_neighbor + '|'

        try:
            item2user_neighbor = nx.neighbors(G, user)
            np.random.shuffle(item2user_neighbor)

            if len(item2user_neighbor) > Max_Num_Neighbor:
                item2user_neighbor = item2user_neighbor[:Max_Num_Neighbor]
        except:
            item2user_neighbor = []

        item2user_neighbor = ','.join([str(x.split('_')[1]) for x in item2user_neighbor])
        line = line + item2user_neighbor

        f.writelines(line)
        f.writelines('\n')
    f.close()


if __name__=="__main__":
    print '=============03: Construct Direct neighbors of NNCF=========='
    relation_train_positive = pd.read_csv('./Data/relation_train_positive.csv')
    relation_train = pd.read_csv('./Data/relation_train.csv')
    relation_test = pd.read_csv('./Data/relation_test.csv')
    relation_train_label = pd.read_csv('./Data/relation_train_label.csv')
    # Max_Num_Neighbor: the maximum number of neighbors is set to 50
    #Max_Num_Neighbor = 50
    Max_Num_Neighbor=int(sys.argv[1])
    print 'max_neighbors:', Max_Num_Neighbor
    G = nx.Graph()
    G.add_edges_from(relation_train_positive.values)
    prepare_vector_element_test(G, relation_test, 'relation_test_vec_element')
    train_original_len = len(relation_train)
    #print train_original_len
    prepare_vector_element_train(G,relation_train,relation_train_label,'relation_train_vec_element_all_data')
    print 'Direct neighbors of NNCF are generated successfully!'
