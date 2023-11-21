# coding:utf-8
'''
**************************************************
@File   ：LSTM -> csv_debug
@IDE    ：PyCharm
@Author ：Tianze Zhang
@Desc   , 
@Date   ：2023/11/19 13:03
**************************************************
'''
import numpy as np
import pandas as pd
data_file_path="backstory.pkl"
df = pd.read_pickle(data_file_path)
titles=df["Title"]
sentences = df["Desc"]
sentences = sentences.tolist()
len_data=len(titles)

for i in range(len_data):
    #print(i,type(sentences[i]),str(sentences[i]))
    sentences[i]="This is the story of [PAWN_nameDef], a "+titles[i]+": "+str(sentences[i])
    # TODO problem at sentence[21], reason: bad sentence truncation

df = pd.DataFrame(sentences)
# Create a custom dataset
# dataset = RimWordDS(sentences, tokenizer)

print(sentences[20])
print(sentences[21])
print(sentences[22])