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
titles = df["Title"].tolist()
titles_train, titles_test = train_test_split(titles, test_size=0.1, random_state=42)
titles_test = titles_test[:50]

model_path = "models/ft1"
print("using device " + str(device))
num_output = 50  # number of output desired

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True).to(device)
# tokenizer = AutoTokenizer.from_pretrained('gpt2-large')
# model = AutoModelForCausalLM.from_pretrained('gpt2-large').to(device)

for title_test in titles_test:
    prompt = "This is the story of [PAWN_nameDef], a" + title_test + ":"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

print(input_ids.get_device())

# sample up to 30 tokens
# torch.manual_seed(11)
list_out = []
sentence_out = []
# ['Today I believe we can finally get rid of discrimination," said Rep. Mark Pocan (D-Wis.).\n\n"Just look at the']
for i in range(num_output):
    outputs = model.generate(input_ids, do_sample=True, max_length=70, temperature=1, top_p=1,
                             repetition_penalty=1)
    # TODO output length
    list_out.append(outputs)

for i in range(num_output):
    generate_sentence = tokenizer.batch_decode(list_out[i], skip_special_tokens=True)
    sentence_out.append(generate_sentence)
df = pd.DataFrame(sentence_out)
df.to_pickle('models/Story_generate/ft1_50_70.pkl')
