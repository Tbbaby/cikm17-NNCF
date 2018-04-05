import pandas as pd
import numpy as np
from sklearn import dummy, metrics, cross_validation, ensemble
import keras.models as kmodels
import keras.layers as klayers
import keras.backend as K
import keras
from keras.layers import Conv1D, MaxPooling1D, Embedding
#from keras import initializations
from math import log
import time
import warnings
warnings.filterwarnings("ignore")
import sys
from keras import optimizers

"""
# nb_layer: the number of MLP layer
Output: the testing evaluation to: hr@10, ndcg@10, loss files
"""

def pad_sequences(X,maxlen):
    if len(X) > maxlen:
        v = np.array(X[:maxlen])
    else:
        v = np.zeros(maxlen)
        for i,x in enumerate(X):
            v[i] = x
    v = v.astype('int')
    return v


def deal_with_line(line,MAX_USER_NEIGHBORS_LEN,MAX_ITEM_NEIGHBORS_LEN):
    line = line.strip()
    label,user_item,user2item_neighbors,item2user_neighbors = line.strip().split('|')
    label = int(label)
    user,item = [int(x) for x in user_item.split(',')]
    try:
        item2user_neighbors = [int(x) for x in item2user_neighbors.split(',')]
    except:
        item2user_neighbors = []
    try:
        user2item_neighbors = [int(x) for x in user2item_neighbors.split(',')]
    except:
        user2item_neighbors = []
    user2item_neighbors_vec = pad_sequences(user2item_neighbors,maxlen=MAX_USER_NEIGHBORS_LEN)#user#to_several_hot(user,user2item_neighbors,n_users)
    item2user_neighbors_vec = pad_sequences(item2user_neighbors,maxlen=MAX_ITEM_NEIGHBORS_LEN)#item#to_several_hot(item,item2user_neighbors,n_items)
    return user , item , user2item_neighbors_vec , item2user_neighbors_vec , label


def get_id_model(n_users, n_items, embedding_dim, nb_layer):
    userid_input = keras.layers.Input(shape=([1]), dtype='int32')
    userid_vec = keras.layers.Flatten()(
        keras.layers.Embedding(input_dim=n_users, output_dim=embedding_dim, name='userid_input_embedding')(
            userid_input))

    movieid_input = keras.layers.Input(shape=([1]), dtype='int32')
    movieid_vec = keras.layers.Flatten()(
        keras.layers.Embedding(input_dim=n_items, output_dim=embedding_dim, name='movieid_input_embedding')(
            movieid_input))

    element_multiple = keras.layers.merge([userid_vec, movieid_vec], mode='mul')
    element_multiple = keras.layers.Dropout(0.5)(element_multiple)

    nn = keras.layers.Dropout(0.5)(keras.layers.Dense(128, activation='relu')(element_multiple))
    nn = keras.layers.normalization.BatchNormalization()(nn)

    for i in range(1, nb_layer):
        nn = keras.layers.Dropout(0.5)(keras.layers.Dense(128, activation='relu')(nn))
        nn = keras.layers.normalization.BatchNormalization()(nn)

    nn = keras.layers.Dense(128, activation='relu')(nn)

    # result = keras.layers.Dense(2, activation='softmax')(nn)

    result = keras.layers.Dense(1, activation='sigmoid')(nn)

    model = kmodels.Model([userid_input, movieid_input], result)
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

    return model


def get_neighbor_model(n_users, n_items, embedding_dim, nb_layer):
    movie_input = keras.layers.Input(shape=(MAX_ITEM_NEIGHBORS_LEN,), dtype='int32')
    movie_vec = keras.layers.Embedding(input_dim=n_items, output_dim=embedding_dim, name='movie_input_embedding')(
        movie_input)
    movie_vec = Conv1D(128, 5, activation='tanh')(movie_vec)
    movie_vec = MaxPooling1D(5)(movie_vec)
    movie_vec = keras.layers.Flatten()(movie_vec)
    movie_vec = keras.layers.Dropout(0.5)(movie_vec)

    user_input = keras.layers.Input(shape=(MAX_USER_NEIGHBORS_LEN,), dtype='int32')
    user_vec = keras.layers.Embedding(input_dim=n_users, output_dim=embedding_dim, name='user_input_embedding')(
        user_input)
    user_vec = Conv1D(128, 5, activation='tanh')(user_vec)
    user_vec = MaxPooling1D(5)(user_vec)
    user_vec = keras.layers.Flatten()(user_vec)
    user_vec = keras.layers.Dropout(0.5)(user_vec)

    input_vecs = keras.layers.merge([movie_vec, user_vec], mode='concat')

    nn = keras.layers.Dropout(0.5)(keras.layers.Dense(128, activation='relu')(input_vecs))
    nn = keras.layers.normalization.BatchNormalization()(nn)

    for i in range(1, nb_layer):
        nn = keras.layers.Dropout(0.5)(keras.layers.Dense(128, activation='relu')(nn))
        nn = keras.layers.normalization.BatchNormalization()(nn)

    nn = keras.layers.Dense(128, activation='relu')(nn)

    # result = keras.layers.Dense(2, activation='softmax')(nn)

    result = keras.layers.Dense(1, activation='sigmoid')(nn)

    model = kmodels.Model([user_input, movie_input], result)
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

    return model


def get_model(n_users, n_items, embedding_dim, nb_layer, MAX_USER_NEIGHBORS_LEN,MAX_ITEM_NEIGHBORS_LEN,LR):
    userid_input = keras.layers.Input(shape=([1]), dtype='int32')
    userid_vec = keras.layers.Flatten()(
        keras.layers.Embedding(input_dim=n_users, output_dim=embedding_dim, name='model_userid_input_embedding')(
            userid_input))

    movieid_input = keras.layers.Input(shape=([1]), dtype='int32')
    movieid_vec = keras.layers.Flatten()(
        keras.layers.Embedding(input_dim=n_items, output_dim=embedding_dim, name='model_movieid_input_embedding')(
            movieid_input))

    element_multiple = keras.layers.merge([userid_vec, movieid_vec], mode='mul')
    element_multiple = keras.layers.Dropout(0.5)(element_multiple)

    movie_input = keras.layers.Input(shape=(MAX_ITEM_NEIGHBORS_LEN,), dtype='int32')
    movie_vec = keras.layers.Embedding(input_dim=n_items, output_dim=embedding_dim, name='model_movie_input_embedding')\
        (movie_input)
    movie_vec = Conv1D(128, 5, activation='tanh')(movie_vec)
    movie_vec = MaxPooling1D(5)(movie_vec)
    movie_vec = keras.layers.Flatten()(movie_vec)
    movie_vec = keras.layers.Dropout(0.5)(movie_vec)

    user_input = keras.layers.Input(shape=(MAX_USER_NEIGHBORS_LEN,), dtype='int32')
    # user_input = keras.layers.Input(shape=([1]),dtype = 'int32')
    # user_input = keras.layers.Input(shape=(n_users,),dtype = 'int32')
    user_vec = keras.layers.Embedding(input_dim=n_users, output_dim=embedding_dim, name='model_user_input_embedding')(
        user_input)
    user_vec = Conv1D(128, 5, activation='tanh')(user_vec)
    user_vec = MaxPooling1D(5)(user_vec)
    user_vec = keras.layers.Flatten()(user_vec)
    user_vec = keras.layers.Dropout(0.5)(user_vec)

    input_vecs = keras.layers.merge([movie_vec, element_multiple, user_vec], mode='concat')

    for i in range(nb_layer):
        input_vecs = keras.layers.Dropout(0.5)(keras.layers.Dense(128, activation='relu')(input_vecs))
        input_vecs = keras.layers.normalization.BatchNormalization()(input_vecs)
    result = keras.layers.Dense(1, activation='sigmoid')(input_vecs)
    model = kmodels.Model([userid_input, movieid_input, user_input, movie_input], result)
    adam=optimizers.Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam,  loss='binary_crossentropy', metrics=['accuracy'])
    #model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

    return model


def Save_List (List, Name):
    File = Name + '.txt'
    pd.DataFrame({Name: List}).to_csv(File, encoding='utf8', header=True, index=False)


def train_model(model, model_type, Epoch,userids,itemids,user_vecs,item_vecs,labels,N,Batchsize):
    hr = 0

    hr_list = []
    ndcg_list = []
    loss_list = []

    for epoch in range(Epoch):
        t1 = time.time()
        if model_type == 'model':
            hist = model.fit([userids, itemids, user_vecs, item_vecs], labels,
                             batch_size=Batchsize,
                             nb_epoch=1,
                             verbose=1)

        elif model_type == 'id_model':

            hist = model.fit([userids, itemids], labels,
                             batch_size=1024,
                             nb_epoch=1,
                             verbose=1)

        elif model_type == 'neighbor_model':

            hist = model.fit([user_vecs, item_vecs], labels,
                             batch_size=1024,
                             nb_epoch=1,
                             verbose=1)

        hr_at_10, model_NDCG = evaluate(model, model_type,MAX_USER_NEIGHBORS_LEN,MAX_ITEM_NEIGHBORS_LEN,N)
        hr_list.append(hr_at_10)
        ndcg_list.append(model_NDCG)
        loss = hist.history['loss'][0]
        loss_list.append(loss)
        t2 = time.time()

        print 'Epoch %d , loss : %.4f , hr : %.4f , ndcg : %.4f , time : %.1f s' % (
        epoch + 1, loss, hr_at_10, model_NDCG, t2 - t1)

        if hr_at_10 > hr:
            json_string = model.to_json()
            open('my_model_architecture_' + str(model_type) + '.json', 'w').write(json_string)
            model.save_weights('my_model_weights_' + str(model_type) + '.h5')
            hr = hr_at_10

    Save_List(hr_list, 'hr_10' + str(model_type))

    Save_List(ndcg_list, 'ndcg_10' + str(model_type))

    Save_List(loss_list, 'loss' + str(model_type))


def deal_with_line_eval(line,MAX_USER_NEIGHBORS_LEN,MAX_ITEM_NEIGHBORS_LEN):
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


def evaluate(model, model_type,MAX_USER_NEIGHBORS_LEN,MAX_ITEM_NEIGHBORS_LEN,N):
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
        user, item, user2item_neighbors_vec, item2user_neighbors_vec = deal_with_line_eval(test,MAX_USER_NEIGHBORS_LEN,MAX_ITEM_NEIGHBORS_LEN)
        test_user_item_pair = str(user) + '_' + str(item)
        userids.append(user)
        itemids.append(item)
        user_item_pair.append(str(user) + '_' + str(item))
        user_vecs.append(user2item_neighbors_vec)
        item_vecs.append(item2user_neighbors_vec)
        uninteract = uninteract_vec_lines[i * N:(i + 1) * N]
        for l in uninteract:
            l = l.strip()
            user, item, user2item_neighbors_vec, item2user_neighbors_vec = deal_with_line_eval(l,MAX_USER_NEIGHBORS_LEN,MAX_ITEM_NEIGHBORS_LEN)
            userids.append(user)
            itemids.append(item)
            user_item_pair.append(str(user) + '_' + str(item))
            user_vecs.append(user2item_neighbors_vec)
            item_vecs.append(item2user_neighbors_vec)

        user_vecs = np.array(user_vecs)
        item_vecs = np.array(item_vecs)
        userids = np.array(userids)
        itemids = np.array(itemids)

        if model_type == 'model':
            pred_prob = model.predict([userids, itemids, user_vecs, item_vecs])
        elif model_type == 'id_model':
            pred_prob = model.predict([userids, itemids])
        elif model_type == 'neighbor_model':
            pred_prob = model.predict([user_vecs, item_vecs])

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

    model_NDCG_at10 = []

    for p in test_pred_loc.location.values:
        if p <= 10:
            model_NDCG_at10.append(1.0 / (log(1 + p) / log(2)))
        else:
            model_NDCG_at10.append(0)

    model_NDCG = np.mean(model_NDCG_at10)

    return model_hr[-1], model_NDCG


if __name__=="__main__":

    print '=======================05: Training model========================'
    '''
    LR=0.01
    embedding_dim = 32
    nb_layer = 1
    nb_epoch =1
    MAX_USER_NEIGHBORS_LEN = 50
    MAX_ITEM_NEIGHBORS_LEN = 50
    N=99
    Batchsize=1024
    '''



    embedding_dim = int(sys.argv[2])
    nb_layer = int(sys.argv[3])
    nb_epoch =int (sys.argv[4])
    MAX_USER_NEIGHBORS_LEN =int(sys.argv[1])
    MAX_ITEM_NEIGHBORS_LEN = int(sys.argv[1])
    N=int(sys.argv[5])
    LR=float(sys.argv[6])
    Batchsize = 1024


    print 'max_neighbors: ', MAX_USER_NEIGHBORS_LEN
    print  'embedding_dim: ',embedding_dim
    print  'nb_layer: ', nb_layer
    print 'nb_epoch: ',nb_epoch
    print 'N: ', N
    print 'LR',LR

    f = open('./Data/relation_train_vec_element_all_data.txt')
    lines = f.readlines()
    user_vecs = []
    item_vecs =[]
    userids = []
    itemids = []
    labels = []
    for line in lines:
        line = line.strip()
        user,item,user2item_neighbors_vec,item2user_neighbors_vec,label = deal_with_line(line,MAX_USER_NEIGHBORS_LEN,MAX_ITEM_NEIGHBORS_LEN)
        user_vecs.append(user2item_neighbors_vec)
        item_vecs.append(item2user_neighbors_vec)
        userids.append(user)
        itemids.append(item)
        labels.append(label)
    user_vecs = np.array(user_vecs)
    item_vecs = np.array(item_vecs)
    userids = np.array(userids)
    itemids = np.array(itemids)
    labels = np.array(labels)
    '''
    print 'user_vecs shape '
    print user_vecs.shape
    print 'item_vecs shape '
    print item_vecs.shape
    '''
    all_item = pd.read_csv('./Data/all_item.csv')
    all_user = pd.read_csv('./Data/all_user.csv')
    n_items = len(all_item)
    n_users = len(all_user)
    model = get_model(n_users, n_items, embedding_dim, nb_layer, MAX_ITEM_NEIGHBORS_LEN, MAX_USER_NEIGHBORS_LEN,LR)
    train_model(model, 'model', nb_epoch, userids, itemids, user_vecs, item_vecs, labels,N,Batchsize)
    print 'Training model successfully!'
    print 'Model saved: my_model_architecture_model.json'
    print 'Model weight saved: my_model_weights_model.h5'






