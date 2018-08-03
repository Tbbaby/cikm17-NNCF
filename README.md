# cikm17-NNCF

Implementation of Neighborhood-based Neural Collaborative Filtering model (NNCF)

Ting Bai et al. "A Neural Collaborative Filtering Model with Interaction-based Neighborhood." Proceedings of the 2017 ACM on Conference on Information and Knowledge Management. ACM, 2017.

## Run the model: python main.py


Parameters:

N_test_negative: the number of negative samples in the testing ranking list

max_neighbors: the maximum neighbors in our algorithm

N_train_negative: the number of negative samples in training

embedding_dim: the output dimension of MLP

nb_layer: the number of layers in MLP

nb_epoch: training epoch

LR: learning rate

## File Description

01: Process input data: data.csv (userid,itemid)

02: Split train & test set and construct graph

03,04: Construct direct neighbors of model (NNCF_direct)

03-1,04-1: Construct community neighbors of model (NNCF_community)

03-2: Construct knn neighbors of model (NNCF_knn)

05: Training model

06: Evaluation of model

The python files are independent to make our project more flexible and extensible. You can tuning parameters and run the corresponding python file that you need.


## Requirement
Python version: 2.7.3

Keras version：2.1.5

Tensorflow: 1.6.0. 

    
## Cite

Please cite our paper if you use this code in your own work:


@inproceedings{bai2017neural,<br>
  title={A neural collaborative filtering model with interaction-based neighborhood},<br>
  author={Bai, Ting and Wen, Ji-Rong and Zhang, Jun and Zhao, Wayne Xin},<br>
  booktitle={Proceedings of the 2017 ACM on Conference on Information and Knowledge Management},<br>
  pages={1979--1982},<br>
  year={2017},<br>
  organization={ACM}<br>
}
