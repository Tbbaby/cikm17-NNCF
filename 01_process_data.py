import pandas as pd
from collections import Counter
import numpy as np

"""
Input data: data.csv(userid, itemid)
"""


if __name__=="__main__":
    print '================01: Statistics of input data================='
    Data = pd.read_csv('./Data/data.csv',names = ['user','item'])
    userid = Data.user
    movieid = Data.item

    rating = abs(np.random.randn(len(Data))*10).astype('int')+1
    timestamp = abs(np.random.randn(len(Data)))

    Pair_Data = pd.DataFrame({'userid':userid,'movieid':movieid,'rating':rating,'timestamp':timestamp})
    Pair_Data = Pair_Data[['userid', 'movieid', 'rating', 'timestamp']]

    # select user
    select_num = 0
    user_count = Counter(Pair_Data.userid.values)
    user_select = [x[0] for x in user_count.items() if x[1] >= select_num ]
    user_select_set = set(user_select)
    tag = [1 if user in user_select_set else 0 for user in Pair_Data.userid.values]
    Pair_Data['tag'] = tag
    Pair_Data_Select = Pair_Data[Pair_Data.tag == 1]

    Pair_Data_Select = Pair_Data_Select.ix[:,:4]

    Pair_Data_Select.to_csv('./Data/data_for_split.csv',index =False)

    print 'number of users: ', len(set(Pair_Data_Select.userid))
    print 'number of items: ', len(set(Pair_Data_Select.movieid))
    print 'number of pairs', len(Pair_Data_Select)
