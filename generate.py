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
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_file_path = "raw_data/backstory.pkl"
df = pd.read_pickle(data_file_path)
#split the df into 2
df_train, df_test = train_test_split(df, test_size=0.1, random_state=44)
df_test=df_test.reset_index()
titles=df_test["Title"]
attributes=df_test["Attribute"]



model_path = "models/ft1"
print("using device " + str(device))

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True).to(device)
#tokenizer = AutoTokenizer.from_pretrained('gpt2')
#model = AutoModelForCausalLM.from_pretrained('gpt2').to(device)

num_output = 10  # number of output desired
ids_ls = []
for i in range(num_output):
    skill_modifiers_str = attributes[i].lower().replace("\t", ", ").replace("-", " -").replace("+", " +").strip(", ")
    prompt = "This is the story of [PAWN_nameDef], a " + titles[i] +": "
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    ids_ls.append(input_ids)

#print(input_ids.get_device())

# sample up to 30 tokens
list_out = []
sentence_out = []
total_loss_test = 0
for i in range(num_output):
    outputs = model.generate(ids_ls[i], do_sample=True, max_length=80, temperature=1, top_p=1,repetition_penalty=1)
    # TODO output length
    list_out.append(outputs)
loss = total_loss_test/50

for i in range(num_output):
    generate_sentence = tokenizer.batch_decode(list_out[i], skip_special_tokens=True)
    sentence_out.append(generate_sentence)
    print(generate_sentence)
df = pd.DataFrame(sentence_out)
df.to_pickle('models/Story_generate/gpt2_50_70.pkl')
