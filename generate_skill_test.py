# coding:utf-8
'''
**************************************************
@File   ：LSTM -> test
@IDE    ：PyCharm
@Author ：Tianze Zhang
@Desc   ,
@Date   ：2023/11/20 14:54
**************************************************
'''
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "models/ft1"
print("using device " + str(device))

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True).to(device)


num_output = 16  # number of output desired
ids_ls = []
for i in range(8):
    prompt="This is the story of[PAWN_nameDef], a gardener with plants "+str(i-8)+":"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    ids_ls.append(input_ids)
for i in range(num_output-8):
    prompt="This is the story of[PAWN_nameDef], a gardener with plants +"+str(i)+":"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    ids_ls.append(input_ids)


list_out = []
sentence_out = []
total_loss_test = 0
for i in range(num_output):
    outputs = model.generate(ids_ls[i], do_sample=True, max_length=90, temperature=1, top_p=1,repetition_penalty=1)
    # TODO output length
    list_out.append(outputs)
loss = total_loss_test/50

for i in range(num_output):
    generate_sentence = tokenizer.batch_decode(list_out[i], skip_special_tokens=True)
    sentence_out.append(generate_sentence)
    print(generate_sentence[0])
df = pd.DataFrame(sentence_out)
df.to_pickle('models/Story_generate/skill_modifier_sample.pkl')
