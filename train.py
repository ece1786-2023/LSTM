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
from torch.optim import SGD, ASGD, Rprop, Adagrad
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm


class RimWordDS(Dataset):
    """
    Define a custom dataset class
    :param sentences: a list a strings that are full sentences for training
    :param tokenizer: a tokenizer to tokenize the sentences
    :param max_length: the max length of each tokenized sentence
    """

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
        count = 0
        for i in range(5):
            count += int(loss_ot[epoch - i] > loss_ot[epoch - i - 1])
        if count >= 3:
            print("Reached plateau in epoch", epoch + 1)

            # if loss_ot[epoch] > loss_ot[epoch - 1] and loss_ot[epoch - 1] > loss_ot[epoch - 2]:
            return True
        else:
            return False


def adjust_learning_rate(lr):
    return lr / 5


def train(max_epochs=30, learning_rate=5e-5, model_name_load="gpt2", model_name_save="models/ft1",
          data_file_path="raw_data/backstory_large.pkl", test_size=0.1, random_state=42, batch_size=8,
          optimizer_name=AdamW):
    """
    Trains a model and saves the model and loss records
    :param max_epochs: the maximum number of epochs the trainer is allowed to run, if early stopping condition is never met
    :param learning_rate: the learning rate
    :param model_name_load: a huggingface model or the path of a previously saved model
    :param model_name_save: the path to save the model
    :param data_file_path: the path of the data file
    :param test_size: proportion of test set
    :param random_state: random state used to train test split the data
    :param batch_size: the batch size
    :param optimizer_name: the optimizer used to optimize the model
    :return none
    """

    # Add special tokens
    token_name = "[PAWN_nameDef]"
    token_possessive = "[PAWN_possessive]"
    token_pronoun = "[PAWN_pronoun]"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device " + str(device))

    tokenizer = AutoTokenizer.from_pretrained(model_name_load)
    # TODO attention mask and the pad token id
    # The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
    # Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
    model = AutoModelForCausalLM.from_pretrained(model_name_load).to(device)

    # data_file_path="backstory.pkl"
    df = pd.read_pickle(data_file_path)
    attributes = df["Attribute"]
    titles = df["Title"]
    sentences = df["Desc"]
    sentences = sentences.tolist()
    len_data = len(titles)

    # generating sentences (training instances)
    for i in range(len_data):
        # print(i,type(sentences[i]),str(sentences[i]))
        skill_modifiers_str = attributes[i].lower().replace("\t", ", ").replace("-", " -").replace("+", " +").strip(
            ", ")
        sentences[i] = "This is the story of [PAWN_nameDef], a " + titles[i] + " with " + skill_modifiers_str + ": " + \
                       sentences[i]
        # print(sentences[i])

    # convert backstory into proper prompt

    sentences_train, sentences_test = train_test_split(sentences, test_size=test_size, random_state=random_state)
    # using a small dataset for development purposes
    # sentences_train, sentences_test = train_test_split(sentences_test, test_size=0.1, random_state=42)

    # Create a custom dataset
    dataset_train = RimWordDS(sentences_train, tokenizer)
    dataset_test = RimWordDS(sentences_test, tokenizer)

    # TODO cross validation?

    # Set up DataLoader
    # dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=True)
    # dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=True)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                                  collate_fn=lambda batch: lm_collate_fn(batch, device))
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True,
                                 collate_fn=lambda batch: lm_collate_fn(batch, device))
    dataloader_train_len = len(dataloader_train)
    dataloader_test_len = len(dataloader_test)

    # Set up optimizer
    optimizer = optimizer_name(model.parameters(), lr=learning_rate)
    # TODO optimizer selection

    loss_ot_train = np.zeros(max_epochs)
    loss_ot_test = np.zeros(max_epochs)
    best_loss = float('inf')

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
        opt_lr = optimizer.param_groups[0]['lr']

        if total_loss_test < best_loss:
            best_loss = total_loss_test

            # save model and tokenizer
            # model.save_pretrained(model_name_save)
            # tokenizer.save_pretrained(model_name_save)
            print('Save best model')

        print(f"Epoch {epoch}: Test loss={total_loss_test} learning_rate={opt_lr}")

        # record loss of the epoch

        if test_loss_plateau(epoch, loss_ot_test):
            # new_lr = adjust_learning_rate(learning_rate)
            # optimizer = optimizer_name(model.parameters(), lr=new_lr)
            break

        loss_ot_train[epoch] = total_loss_train / dataloader_train_len
        loss_ot_test[epoch] = total_loss_test

    # save loss records
    train_save_path = "loss_record/lr_" + str(learning_rate) + "_bs_" + str(
        batch_size) + "_" + optimizer_name.__name__ + "_train"
    test_save_path = "loss_record/lr_" + str(learning_rate) + "_bs_" + str(
        batch_size) + "_" + optimizer_name.__name__ + "_test"
    np.save(train_save_path, loss_ot_train)
    np.save(test_save_path, loss_ot_test)


if __name__ == '__main__':
    train(max_epochs=30,
          learning_rate=5e-6,
          model_name_load="gpt2",
          data_file_path="raw_data/backstory_large.pkl",
          batch_size=16,
          optimizer_name=AdamW)
