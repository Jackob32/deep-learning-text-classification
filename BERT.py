import logging
import time
from platform import python_version
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable

print("python version==%s" % python_version())
print("pandas==%s" % pd.__version__)
print("numpy==%s" % np.__version__)
print("torch==%s" % torch.__version__)
print("sklearn==%s" % sklearn.__version__)
print("transformers==%s" % transformers.__version__)
print("matplotlib==%s" % matplotlib.__version__)

df = pd.read_csv('data/train.csv')

np.random.seed(42)
df = df.sample(frac=1)
df = df.reset_index(drop=True)

df.head()

print(df.comment_text[0])

df_train = df[:1000].reset_index(drop=True)
df_val = df[1000:1100].reset_index(drop=True)
df_test = df[1100:1300].reset_index(drop=True)

pretrained_weights='distilbert-base-uncased'
# Load pretrained model/tokenizer

# Load pretrained model/tokenizer
tokenizer = transformers.DistilBertTokenizer.from_pretrained(pretrained_weights)
model = transformers.DistilBertModel.from_pretrained(pretrained_weights)

max_seq = 100

def tokenize_text(df, max_seq):
    return [tokenizer.encode(text[:max_seq], add_special_tokens=True) for text in df.comment_text.values]

def pad_text(tokenized_text, max_seq):
    return np.array([el + [0] * (max_seq - len(el)) for el in tokenized_text])

def tokenize_and_pad_text(df, max_seq):
    tokenized_text = tokenize_text(df, max_seq)
    padded_text = pad_text(tokenized_text, max_seq)
    return torch.tensor(padded_text, dtype=torch.long)

def targets_to_tensor(df, target_columns):
    return torch.tensor(df[target_columns].values, dtype=torch.float32)



input_ids = torch.tensor([tokenizer.encode("Here is some text to encode",
                                               add_special_tokens=True)])  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.

with torch.no_grad():
    print(input_ids)
    last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples

    print(last_hidden_states)

train_indices = tokenize_and_pad_text(df_train, max_seq)
val_indices = tokenize_and_pad_text(df_val, max_seq)
test_indices = tokenize_and_pad_text(df_test, max_seq)

print(train_indices)

with torch.no_grad():
    x_train = model(train_indices)[0]
    x_val = model(val_indices)[0]
    x_test = model(test_indices)[0]

y_train = targets_to_tensor(df_train, target_columns)
y_val = targets_to_tensor(df_val, target_columns)
y_test = targets_to_tensor(df_test, target_columns)


