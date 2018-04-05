import pandas as pd
import numpy as np
import networkx as nx
from collections import Counter
import networkx as nx
import community
import os 
import scipy as sp
from scipy.sparse import csr_matrix
import time
import sys

"""
To construct community neighbors of NNCF
Output files: 
relation_test_vec_element.csv
relation_train_vec_element_all_data.csv
"""


def Cos(v1,M):
    return np.dot(v1,M.T)/(np.linalg.norm(v1)*np.linalg.norm(M.T,axis=0))


def get_neighbor(node,M,k=50):
    kind,i = node.split('_')
    i = int(i)
    if kind == 'user':
        cos_score = Cos(M[i,:],M)
        cos_index_zip = zip(range(len(cos_score)),cos_score)
        cos_index_zip.sort(key= lambda x : x[1],reverse=True)
        neighbor = [x[0] for x in cos_index_zip[:k]]
        neighbor = [kind+'_'+str(x) for x in neighbor]
    if kind == 'item':
        M = M.T
        cos_score = Cos(M[i,:],M)
        cos_index_zip = zip(range(len(cos_score)),cos_score)
        cos_index_zip.sort(key= lambda x : x[1],reverse=True)
        neighbor = [x[0] for x in cos_index_zip[:k]]
        neighbor = [kind+'_'+str(x) for x in neighbor]
    return neighbor


def prepare_vector_element_test(relation, filename, user_neighbor_dict, item_neighbor_dict):
    f = open('./Data/' + filename + '.txt', 'wb')
    for r in relation.values:
        line = ''
        user, item = r
        line = line + user.split('_')[1] + ','
        line = line + item.split('_')[1] + '|'
        try:
            item2user_neighbor = user_neighbor_dict[user]
            np.random.shuffle(item2user_neighbor)
            if len(item2user_neighbor) > Max_Num_Neighbor:
                item2user_neighbor = item2user_neighbor[:Max_Num_Neighbor]
        except:
            item2user_neighbor = []

        item2user_neighbor = ','.join([str(x.split('_')[1]) for x in item2user_neighbor])
        line = line + item2user_neighbor + '|'
        try:
            user2item_neighbor = item_neighbor_dict[item]
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


def prepare_vector_element_train(relation, label, filename, user_neighbor_dict, item_neighbor_dict):
    f = open('./Data/' + filename + '.txt', 'wb')
    for r, l in zip(relation.values, label.values):
        line = str(l[0]) + '|'
        user, item = r
        line = line + user.split('_')[1] + ','
        line = line + item.split('_')[1] + '|'
        try:
            item2user_neighbor = user_neighbor_dict[user]
            np.random.shuffle(item2user_neighbor)
            if len(item2user_neighbor) > Max_Num_Neighbor:
                item2user_neighbor = item2user_neighbor[:Max_Num_Neighbor]
        except:
            item2user_neighbor = []

        item2user_neighbor = ','.join([str(x.split('_')[1]) for x in item2user_neighbor])
        line = line + item2user_neighbor + '|'
        try:
            user2item_neighbor = item_neighbor_dict[item]
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
    print '===========03-2: Construct KNN neighbors of NNCF==========='
    #Max_Num_Neighbor = 50
    #N = 99
    Max_Num_Neighbor =int(sys.argv[2])
    N = int(sys.argv[1])
    relation_train_positive = pd.read_csv('./Data/relation_train_positive.csv')
    relation_train = pd.read_csv('./Data/relation_train.csv')
    relation_test = pd.read_csv('./Data/relation_test.csv')
    relation_train_label = pd.read_csv('./Data/relation_train_label.csv')

    user_index = [int(x.split('_')[1]) for x in relation_train_positive.user]
    item_index = [int(x.split('_')[1]) for x in relation_train_positive.item]
    M = np.zeros([max(user_index) + 1, max(item_index) + 1])
    for u, i in zip(user_index, item_index):
        M[u, i] = 1

    user_node = []
    user_neighbor = []
    item_node = []
    item_neighbor = []

    i = 0
    for u in list(set(relation_train_positive.user)):
        i = i + 1
        t1 = time.time()
        user_node.append(u)
        user_neighbor.append(get_neighbor(u, M))
        t2 = time.time()
        #print 'user num :  %d , cost %.2f s' % (i, t2 - t1)
    user_neighbor_dict = dict(zip(user_node, user_neighbor))

    j = 0
    for i in list(set(relation_train_positive.item)):
        j = j + 1
        t1 = time.time()
        item_node.append(i)
        item_neighbor.append(get_neighbor(i, M))
        t2 = time.time()
        #print 'item num :  %d , cost %.2f s' % (j, t2 - t1)
    item_neighbor_dict = dict(zip(item_node, item_neighbor))

    prepare_vector_element_test(relation_test, 'relation_test_vec_element', user_neighbor_dict, item_neighbor_dict)

    prepare_vector_element_train(relation_train, relation_train_label, 'relation_train_vec_element_all_data',
                                 user_neighbor_dict, item_neighbor_dict)

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
        interact_item = nx.neighbors(G_all, user)
        un_interact_item = list(set(all_item) - set(interact_item))
        np.random.shuffle(un_interact_item)
        if len(un_interact_item) < N:
            un_interact_item = un_interact_item * (int(N / len(un_interact_item)) + 1)
        un_interact_item_choose = un_interact_item[:N]
        uninteract_user_list += [user] * N
        uninteract_item_list += un_interact_item_choose

    uninteract = pd.DataFrame(np.array([uninteract_user_list, uninteract_item_list]).T, columns=['user', 'item'])

    prepare_vector_element_test(relation_test, 'relation_test_vec_element', user_neighbor_dict, item_neighbor_dict)
    prepare_vector_element_test(uninteract, 'uninteract_vec_element', user_neighbor_dict, item_neighbor_dict)
    print 'KNN neighbors of NNCF are generated successfully!'