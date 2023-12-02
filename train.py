# coding:utf-8
'''
**************************************************
@File   ：LSTM -> train
@IDE    ：PyCharm
@Author ：Tianze Zhang
@Desc   , 
@Date   ：2023/11/19 19:28
**************************************************
'''
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm
import matplotlib.pyplot as plt


# Define a custom dataset class for your sentences
class RimWordDS(Dataset):
    def __init__(self, sentences, tokenizer, max_length=128):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = str(self.sentences[idx])
        inputs = self.tokenizer(sentence, return_tensors="pt", max_length=self.max_length, truncation=True)
        return inputs


def lm_collate_fn(batch, device):
    x = [item.data['input_ids'] for item in batch]  # List (len B) of varying lengths
    y = [item.data['attention_mask'] for item in batch]  # List (len B) of the same lengths as x
    # maxlen = max([len(s) for s in x])
    maxlen = max([s.shape[1] for s in x])

    padded_x, padded_y = [], []
    for sx, sy in zip(x, y):
        padded_x.append(torch.cat([sx.squeeze(), torch.ones(maxlen - sx.shape[1])]))
        padded_y.append(torch.cat([sy.squeeze(), torch.ones(maxlen - sy.shape[1])]))
    for i in range(len(batch)):
        batch[i].data['input_ids'] = padded_x[i].reshape(1, -1)
        batch[i].data['attention_mask'] = padded_y[i].reshape(1, -1)
    return torch.stack(padded_x).long().to(device), torch.stack(padded_y).long().to(device)
    # return torch.stack(batch, 0)


def test_loss_plateau(epoch, loss_ot):
    """
    Retruns True if there has been 2 epochs where test loss increases after the last 4 epochs
    :param epoch: the number of finished training epochs-1
    :param loss_ot: array containing test loss overtime
    """
    # todo tune the plateau parameters 4 and 2
    if epoch < 5:
        return False
    else:
        count=0
        for i in range(5):
            count += int(loss_ot[epoch - i] > loss_ot[epoch - i - 1])
        if count>=3:
            print("Reached plateau in epoch",epoch+1)
            return True
        else: return False


# Add special tokens
token_name = "[PAWN_nameDef]"
token_possessive = "[PAWN_possessive]"
token_pronoun = "[PAWN_pronoun]"

# Fine-tuning parameters
max_epochs = 30  # the maximum number of epochs the trainer is allowed to run, if early stopping condition is never met
learning_rate = 5e-6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device " + str(device))

# the model to load from
model_name_load = "gpt2"  # get a fresh gpt-2 model from hugging face
# model_name_load="ft1" #load a previously saved model, fine-tune 1

# the model to save as
model_name_save = "models/ft1"  # get a fresh gpt-2 model from hugging face

data_file_path = "raw_data/backstory.pkl"

tokenizer = AutoTokenizer.from_pretrained(model_name_load)
# TODO attention mask and the pad token id
# The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
# Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
model = AutoModelForCausalLM.from_pretrained(model_name_load).to(device)

# data_file_path="backstory.pkl"
df = pd.read_pickle(data_file_path)
attributes=df["Attribute"]
titles = df["Title"]
sentences = df["Desc"]
sentences = sentences.tolist()
len_data = len(titles)

# generating sentences (training instances)
for i in range(len_data):
    # print(i,type(sentences[i]),str(sentences[i]))
    skill_modifiers_str=attributes[i].lower().replace("\t", ", ").replace("-", " -").replace("+", " +").strip(", ")
    sentences[i] = "This is the story of [PAWN_nameDef], a " + titles[i] + " with "+skill_modifiers_str+": " + sentences[i]
    #print(sentences[i])

# convert backstory into proper prompt


sentences_train, sentences_test = train_test_split(sentences, test_size=0.1, random_state=42)
# using a small dataset for development purposes
#sentences_train, sentences_test = train_test_split(sentences_test, test_size=0.1, random_state=42)

# Create a custom dataset
dataset_train = RimWordDS(sentences_train, tokenizer)
dataset_test = RimWordDS(sentences_test, tokenizer)

# TODO cross validation?

# Set up DataLoader
# dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=True)
# dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=True)
dataloader_train = DataLoader(dataset_train, batch_size=8, shuffle=True, collate_fn=lambda batch: lm_collate_fn(batch, device))
dataloader_test = DataLoader(dataset_test, batch_size=8, shuffle=True, collate_fn=lambda batch: lm_collate_fn(batch, device))
# TODO padding to allow batching
dataloader_train_len = len(dataloader_train)
dataloader_test_len = len(dataloader_test)

# Set up optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate)
# TODO optimizer selection

loss_ot_train = np.zeros(max_epochs)
loss_ot_test = np.zeros(max_epochs)

# Fine-tune loop
model.train()
for epoch in range(max_epochs):
    total_loss_train = 0
    total_loss_test = 0

    progress_bar = tqdm(dataloader_train, desc=f"Epoch {epoch}")

    for batch in progress_bar:
        # Move batch to device
        batch = {"input_ids": batch[0], "attention_mask": batch[1]}
        # batch = {key: value[0].to(device) for key, value in batch.items()}

        # Forward pass
        outputs = model(**batch, labels=batch["input_ids"])
        loss = outputs.loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss_train += loss.item()
        # Update progress bar
        progress_bar.set_postfix({"Loss": total_loss_train / dataloader_train_len})

    for batch_idx, batch in enumerate(dataloader_test, 1):
        # Move batch to device
        # batch = {key: value[0].to(device) for key, value in batch.items()}
        batch = {"input_ids": batch[0], "attention_mask": batch[1]}

        # Forward pass
        outputs = model(**batch, labels=batch["input_ids"])
        loss = outputs.loss

        total_loss_test += loss.item()
    total_loss_test /= dataloader_test_len
    print(f"Epoch {epoch}: Test loss={total_loss_test}")

    # record loss of the epoch
    loss_ot_train[epoch] = total_loss_train / dataloader_train_len
    loss_ot_test[epoch] = total_loss_test

    if test_loss_plateau(epoch, loss_ot_test):
        max_epochs = epoch + 1
        break

fig, splot = plt.subplots(1)
domain = np.arange(max_epochs)
splot.plot(domain, loss_ot_train[0:max_epochs], 'g', label="train")
splot.plot(domain, loss_ot_test[0:max_epochs], 'r', label="test")
np.save("loss_record/loss_ot_train",loss_ot_train[0:max_epochs])
np.save("loss_record/loss_ot_test",loss_ot_test[0:max_epochs])
plt.xlim([0, max_epochs-1])
# plt.ylim([0, np.ceil(loss_ot_train[0])])
# plt.ylim([0, 3])#doesn't work with log scale
plt.yscale("log")
splot.legend()
splot.title.set_text("Loss over time")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig("figure/loss overtime1.png")

model.save_pretrained(model_name_save)
tokenizer.save_pretrained(model_name_save)