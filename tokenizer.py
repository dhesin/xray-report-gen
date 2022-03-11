import numpy as np
import pandas as pd
import pickle
import logging
from utils import set_seed
import os
from tokenizers import BertWordPieceTokenizer
import re

set_seed(42)
data_path = "/home/desin/CS224N/data/chestXray/"
data_path_mimic = "/home/desin/CS224N/data/mimic_cxr/"



import regex as re
def basicPreprocess(text):
  try:
    processed_text = text.lower()
    #processed_text = re.findall(r"[\w']+|[.,!?;]", processed_text)
    processed_text = processed_text.replace("\n", " ")
    processed_text = re.sub(r'\W +', ' ', processed_text)
  except Exception as e:
    print("Exception:",e,",on text:", text)
    return None
  return processed_text



#################################
# generate vocabulary
#################################
df = pd.read_csv(data_path+"uid_report_projection_label.csv")
df = df[['uid','findings','filename']]
df = df.rename(columns={'uid':'id', 'findings':'report', 'filename':'image_path'})

#df_mimic = pd.read_csv(data_path_mimic+"mimic-cxr-reports.csv")
#df_mimic = df_mimic[['id','report','image_path']]

# combine
#df = pd.concat([df, df_mimic])
#print("size of concated dataframe:", len(df))


df["report"] = df["report"].apply(lambda x: basicPreprocess(x))
df = df.drop_duplicates(subset=['report'])


df = df[["report"]]

df.to_csv("./ui_report_only.csv", index=False, header=None)

print(df)

# Initialize an empty BERT tokenizer
tokenizer = BertWordPieceTokenizer(lowercase=True)

# prepare text files to train vocab on them
files = ['./ui_report_only.csv']


vocab_size=4000
# Customize training
tokenizer.train(
    files=files, 
    vocab_size=vocab_size, 
    min_frequency=5, 
    show_progress=True,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    limit_alphabet=500,
    wordpieces_prefix='##',
    )

# save the vocab
tokenizer.save_model('./', 'bert-vocab-ui')

vocab = 'bert-vocab-ui-vocab.txt'
tokenizer = BertWordPieceTokenizer(vocab)
print("Vocab Size:", tokenizer.get_vocab_size())

# test the tokenizer with some text
print(str(df.iloc[10]['report']))
encoded = tokenizer.encode(str(df.iloc[10]['report']))
print(encoded.tokens)
print(encoded.ids)
print(tokenizer)
print(tokenizer.decode(encoded.ids))
