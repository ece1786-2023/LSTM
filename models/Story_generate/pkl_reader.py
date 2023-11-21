# coding:utf-8
'''
**************************************************
@File   ：LSTM -> pkl_reader
@IDE    ：PyCharm
@Author ：Tianze Zhang
@Desc   , 
@Date   ：2023/11/20 19:10
**************************************************
'''
import pandas as pd

data_file_path = "./gpt2-large_50_70.pkl"
df = pd.read_pickle(data_file_path)
print(df)
