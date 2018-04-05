import pandas as pd
import numpy as np
from sklearn import dummy, metrics, cross_validation, ensemble
import keras.models as kmodels
import keras.layers as klayers
import keras.backend as K
import keras
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras import initializers
from math import log
import time
from collections import Counter
from keras.models import model_from_json
import warnings
warnings.filterwarnings("ignore")
import sys


"""
Evaluate our model
Output files:
HR@1-10: HR_result.txt
NDCG@1-10: NDCG_result.txt
"""


def pad_sequences(X, maxlen):
    if len(X) > maxlen:
        v = np.array(X[:maxlen])
    else:
        v = np.zeros(maxlen)
        for i, x in enumerate(X):
            v[i] = x
    v = v.astype('int')
    return v


def deal_with_line(line, MAX_USER_NEIGHBORS_LEN, MAX_ITEM_NEIGHBORS_LEN):
    line = line.strip()
    user_item, user2item_neighbors, item2user_neighbors = line.strip().split('|')
    user, item = [int(x) for x in user_item.split(',')]
    try:
        item2user_neighbors = [int(x) for x in item2user_neighbors.split(',')]
    except:
        item2user_neighbors = []
    try:
        user2item_neighbors = [int(x) for x in user2item_neighbors.split(',')]
    except:
        user2item_neighbors = []
    user2item_neighbors_vec = pad_sequences(user2item_neighbors, maxlen=MAX_USER_NEIGHBORS_LEN)
    item2user_neighbors_vec = pad_sequences(item2user_neighbors, maxlen=MAX_ITEM_NEIGHBORS_LEN)
    return user, item, user2item_neighbors_vec, item2user_neighbors_vec


def evaluate(model,MAX_USER_NEIGHBORS_LEN,MAX_ITEM_NEIGHBORS_LEN):
    f = open('./Data/relation_test_vec_element.txt', 'r')
    test_vec_lines = f.readlines()
    f.close()
    f = open('./Data/uninteract_vec_element.txt', 'r')
    uninteract_vec_lines = f.readlines()
    f.close()
    test_len = len(test_vec_lines)
    test_index_list = []
    baseline_index_list = []
    for i in range(test_len):
        user_vecs = []
        item_vecs = []
        userids = []
        itemids = []
        user_item_pair = []
        test = test_vec_lines[i].strip()
        user, item, user2item_neighbors_vec, item2user_neighbors_vec = deal_with_line(test,MAX_USER_NEIGHBORS_LEN,MAX_ITEM_NEIGHBORS_LEN)
        test_user_item_pair = str(user) + '_' + str(item)
        userids.append(user)
        itemids.append(item)
        user_item_pair.append(str(user) + '_' + str(item))
        user_vecs.append(user2item_neighbors_vec)
        item_vecs.append(item2user_neighbors_vec)
        uninteract = uninteract_vec_lines[i * 99:(i + 1) * 99]
        for l in uninteract:
            l = l.strip()
            user, item, user2item_neighbors_vec, item2user_neighbors_vec = deal_with_line(l,MAX_USER_NEIGHBORS_LEN,MAX_ITEM_NEIGHBORS_LEN)
            userids.append(user)
            itemids.append(item)
            user_item_pair.append(str(user) + '_' + str(item))
            user_vecs.append(user2item_neighbors_vec)
            item_vecs.append(item2user_neighbors_vec)

        user_vecs = np.array(user_vecs)
        item_vecs = np.array(item_vecs)
        userids = np.array(userids)
        itemids = np.array(itemids)

        pred_prob = model.predict([userids, itemids, user_vecs, item_vecs])

        Z = zip(user_item_pair, pred_prob)
        Z.sort(key=lambda x: x[1], reverse=True)
        pair_list, _ = zip(*Z)
        test_index = pair_list.index(test_user_item_pair) + 1

        test_index_list.append(test_index)

    test_pred_loc = pd.DataFrame({'location': test_index_list})

    model_hr = []

    for n in range(1, 11):
        model_pred_n = [1 if x <= n else 0 for x in test_pred_loc.location.values]
        model_hr.append(sum(model_pred_n) * 1.0 / len(test_pred_loc))

    model_ndcg = []

    for n in range(1, 11):

        model_NDCG_at10 = []
        for p in test_pred_loc.location.values:
            if p <= n:
                model_NDCG_at10.append(1.0 / (log(1 + p) / log(2)))
            else:
                model_NDCG_at10.append(0)

        model_NDCG = np.mean(model_NDCG_at10)
        model_ndcg.append(model_NDCG)
    return model_hr, model_ndcg


if __name__=="__main__":
    print '====================06: Evaluation matrix===================='
    model = model_from_json(open('my_model_architecture_model.json').read())
    model.load_weights('my_model_weights_model.h5')
    #MAX_USER_NEIGHBORS_LEN = 50
    #MAX_ITEM_NEIGHBORS_LEN = 50
    MAX_USER_NEIGHBORS_LEN = int(sys.argv[1])
    MAX_ITEM_NEIGHBORS_LEN = int (sys.argv[1])
    print 'MAX_USER_NEIGHBORS_LEN: ',MAX_USER_NEIGHBORS_LEN
    model_hr, model_ndcg = evaluate(model,MAX_USER_NEIGHBORS_LEN,MAX_ITEM_NEIGHBORS_LEN)
    print 'HR: ', model_hr
    print 'NDCG: ', model_ndcg
    fhr=open('./HR_result.txt','w')
    fndcg=open('./NDCG_result.txt','w')
    for ihr in model_hr:
        fhr.write(str(ihr))
        fhr.write('\n')
    fhr.close()
    for indcg in model_ndcg:
        fndcg.write(str(indcg))
        fndcg.write('\n')
    fndcg.close()
    print 'HR_result.txt is saved successfully!'
    print 'NDCG_result.txt is saved successfully!'
    print '======================Over==========================='
