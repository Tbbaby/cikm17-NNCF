import pandas as pd
import numpy as np
import networkx as nx
from collections import Counter
import itertools
import sys
"""
# output files like below
relation_train_positive.csv
relation_test.csv
all_user.csv
all_item.csv
relation_train.csv
relation_train_label.csv
"""


def Save_DataFrame_csv(DF,File_Name):
    File = File_Name + '.csv'
    DF.to_csv(File, encoding='utf8', header=True, index=False)


if __name__=="__main__":
    print '=======02: split train & test set and construct graph========'
    # the negative samples in training
    #num_of_negitive = 4
    num_of_negitive=int(sys.argv[1])
    print 'the number of negative samples in training: ', num_of_negitive
    ratings = pd.read_csv('./Data/data_for_split.csv')
    ratings.movieid = ratings.movieid.astype('category')
    ratings.userid = ratings.userid.astype('category')
    ratings['movieid_new'] = ratings.movieid.cat.codes.values
    ratings['userid_new'] = ratings.userid.cat.codes.values
    ratings = ratings.sort_values(['userid_new','timestamp'])
    ratings.index = range(len(ratings))
    ratings['graph_userid'] = ['user_'+str(userid_new) for userid_new in ratings.userid_new]
    ratings['graph_itemid'] = ['item_'+str(movieid_new) for movieid_new in ratings.movieid_new]

    # prepare the train & test set and construct graph
    ratings_values = ratings.values
    test = []
    train_edge_user = []
    train_edge_item = []
    start = 0
    user = ratings_values[0, -2]
    for i in range(1, len(ratings)):
        new_user = ratings_values[i, -2]
        if new_user == user:
            pass
        else:
            user = ratings_values[i, -2]
            test.append(ratings_values[i-1, -2:])
            train_edge = ratings_values[start:i-1, -2:]
            train_edge_user += list(train_edge[:, 0])
            train_edge_item += list(train_edge[:, 1])
            start = i
    test.append(ratings_values[i, -2:])
    train_edge = ratings_values[start:i, -2:]
    train_edge_user += list(train_edge[:, 0])
    train_edge_item += list(train_edge[:, 1])
    relation_test = pd.DataFrame(test, columns=['user', 'item'])
    relation_train_positive = pd.DataFrame(np.array([train_edge_user, train_edge_item]).T, columns=['user', 'item'])
    Save_DataFrame_csv(relation_train_positive,'./Data/relation_train_positive')
    Save_DataFrame_csv(relation_test,'./Data/relation_test')

    all_user = list(set(ratings.graph_userid))
    all_item = list(set(ratings.graph_itemid))

    pd.DataFrame(all_user, columns=['user']).to_csv('./Data/all_user.csv', header=True, index=False)
    pd.DataFrame(all_item, columns=['item']).to_csv('./Data/all_item.csv', header=True, index=False)

    all_pairs = list(itertools.product(all_user,all_item))
    # print 'all pairs'
    positive_pairs = [(x[0], x[1]) for x in ratings[['graph_userid', 'graph_itemid']].values]
    negative_pairs = list(set(all_pairs) - set(positive_pairs))
    len_positive = len(relation_train_positive)
    len_negative = len_positive * num_of_negitive
    np.random.shuffle(negative_pairs)
    relation_train_negative = pd.DataFrame(negative_pairs[:len_negative], columns=['user', 'item'])
    relation_train = pd.concat([relation_train_positive,relation_train_negative])
    relation_train.index = range(len(relation_train))
    label = [1] * len(relation_train_positive) + [0] * len(relation_train_negative)
    relation_train_label = pd.DataFrame(label, columns=['label'])

    # need to shuffle this dataframe !!!

    index = range(len(relation_train))
    np.random.shuffle(index)

    relation_train = relation_train.ix[index,:]
    relation_train_label = relation_train_label.ix[index,:]

    Save_DataFrame_csv(relation_train, './Data/relation_train')
    Save_DataFrame_csv(relation_train_label, './Data/relation_train_label')

    print 'training samples: ', len(relation_train)
    print 'positive training samples:', len(relation_train_positive)
