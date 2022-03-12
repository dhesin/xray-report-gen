import numpy as np
import pandas as pd
import pickle
from dataset import get_train_val_df, generate_vocabulary
import logging
from utils import set_seed
import os

set_seed(42)
data_path = "/home/desin/CS224N/data/chestXray/"
data_path_mimic = "/home/desin/CS224N/data/mimic_cxr/"


TRAIN_SIZE = 0.8

#################################
# generate vocabulary
#################################
df = pd.read_csv(data_path+"uid_report_projection_label.csv")
df = df[['uid','findings','filename',\
                                'No Finding','Enlarged Cardiomediastinum','Cardiomegaly','Lung Lesion', \
                                'Lung Opacity','Edema','Consolidation','Pneumonia','Atelectasis',\
                                'Pneumothorax','Pleural Effusion','Pleural Other','Fracture','Support Devices']]
df = df.rename(columns={'uid':'id', 'findings':'report', 'filename':'image_path'})
df['image_path'] = df['image_path'].apply(lambda x: os.path.join(data_path+"/images/images_normalized/", x))

#df_mimic = pd.read_csv(data_path_mimic+"mimic-cxr-reports.csv")
#df_mimic = df_mimic[['id','report','image_path']]
#df_mimic['image_path'] = df_mimic['image_path'].apply(lambda x: os.path.join(data_path_mimic+"/images/", eval(x)[0]))

# combine
#df = pd.concat([df, df_mimic])
#print("size of concated dataframe:", len(df))
df = df.dropna(subset=['image_path'])
df = df.drop_duplicates(subset=['report'])
df['report'] = df['report'].apply(lambda x: x.replace('\n',' '))
df.to_csv("ui_reports.csv")


word_2_id, id_2_word = generate_vocabulary(df)
vocab_size = len(word_2_id)
assert(len(id_2_word) == len(word_2_id))
print(word_2_id)
print("vocabulary size mimic and ui:", len(id_2_word))



##################################
# generate train/validation sets
#################################
#df = pd.read_csv(data_path+"uid_report_projection_label.csv")
train_df, val_df = get_train_val_df(df, TRAIN_SIZE)
print(f'There are {len(train_df) :,} samples for training, and {len(val_df) :,} samples for validation testing')

db_vocab = {"word_2_id":word_2_id, "id_2_word":id_2_word}
with open("./db_vocab.pkl", "wb") as cache:
    pickle.dump(db_vocab, cache)


db_dataset = {"train_df":train_df, "val_df":val_df}
with open("./db_datasets.pkl", "wb") as cache:
    pickle.dump(db_dataset, cache)







