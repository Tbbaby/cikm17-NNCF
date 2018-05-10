import sys
from subprocess import call
import os

"""
three models:
model_direct
model_community
model_knn
"""


def model_direct():
    print 'Enter the parameters: '
    print 'the number of negative samples in the testing ranking list'
    v1 = raw_input()
    print 'max_neighbors'
    v2 = raw_input()
    print 'the  number of negative samples in training'
    v3 = raw_input()
    print 'the output dimension of MLP'
    v4 = raw_input()
    print 'the number of layers in MLP'
    v5 = raw_input()
    print 'training epoch'
    v6 = raw_input()
    print 'learning rate'
    v7=raw_input()
    print '====================Direct model begin!======================'
    """
    v1: N_test_negative
    v2: max_neighbors
    v3: N_train_negative
    v4: embedding_dim
    v5: nb_layer
    v6: nb_epoch
    v7: LR
    """
    call(["python", "01_process_data.py"])
    call(["python", "02_split_train_test_relation.py", v3])
    call(["python", "03_prepare_vec_element.py", v2])
    call(["python", "04_construct_evaluate_vec.py", v1, v2])
    call(["python", "05_Train_Model.py", v2, v4, v5, v6,v1,v7])
    call(["python", "06_final_model_evalution.py", v2,v1])


def model_community():
    print 'Enter the parameters: '
    print 'the number of negative samples in the testing ranking list'
    v1 = raw_input()
    print 'max_neighbors'
    v2 = raw_input()
    print 'the  number of negative samples in training'
    v3 = raw_input()
    print 'the output dimension of MLP'
    v4 = raw_input()
    print 'the number of layers in MLP'
    v5 = raw_input()
    print 'training epoch'
    v6 = raw_input()
    print 'the partition parameter of community'
    v7 = raw_input()
    print 'learning rate'
    v8 = raw_input()

    print '==================Community model begin!==================='
    """
    v1: N_test_negative
    v2: max_neighbors
    v3: N_train_negative
    v4: embedding_dim
    v5: nb_layer
    v6: nb_epoch
    v7: resolution
    """
    call(["python", "01_process_data.py"])
    call(["python", "02_split_train_test_relation.py", v3])
    call(["python", "03-1_prepare_vec_element_community.py", v2, v7])
    call(["python", "04-1_construct_evaluate_vec_community.py", v1, v2, v7])
    call(["python", "05_Train_Model.py", v2, v4, v5, v6, v1,v8])
    call(["python", "06_final_model_evalution.py", v2,v1])


def model_knn():
    print 'Enter the parameters: '
    print 'the number of negative samples in the testing ranking list'
    v1 = raw_input()
    print 'max_neighbors'
    v2 = raw_input()
    print 'the  number of negative samples in training'
    v3 = raw_input()
    print 'the output dimension of MLP'
    v4 = raw_input()
    print 'the number of layers in MLP'
    v5 = raw_input()
    print 'training epoch'
    v6 = raw_input()
    print  'learning rate'
    v7 = raw_input()
    print '====================KNN model begin!======================'
    """
    v1: N_test_negative
    v2: max_neighbors
    v3: N_train_negative
    v4: embedding_dim
    v5: nb_layer
    v6: nb_epoch
    """
    call(["python", "01_process_data.py"])
    call(["python", "02_split_train_test_relation.py", v3])
    call(["python", "03-2_prepare_vec_element_nn.py", v1, v2])
    call(["python", "05_Train_Model.py", v2, v4, v5, v6, v1,v7])
    call(["python", "06_final_model_evalution.py", v2,v1])


if __name__ == '__main__':
    #main.py(N_test_negative, max_neighbors, N_train_negative, embedding_dim, nb_layer, nb_epoch)
    print '===================Select a model (Enter the number)==================='
    print '                         1: NNCF_KNN                         '
    print '                         2: NNCF_Direct                      '
    print '                         3: NNCF_Community                   '
    print '======================================================================'
    num= raw_input()
    if num=='1':
        model_knn()
    if num=='2':
        model_direct()
    if num=='3':
        model_community()





