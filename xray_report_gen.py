import numpy as np
import pandas as pd
import torchvision
import torch
from dataset import chestXRayDataset, get_train_val_df, generate_vocabulary
from torch.utils.data import Dataset,  DataLoader
import logging
from utils import set_seed, sample
from model import ImageEncoderReportDecoder, ImageEncoderReportDecoderConfig
from trainer import Trainer, TrainerConfig

logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

set_seed(42)


##################################
# load resnet18 for image encoding
##################################
img_enc = torch.hub.load('pytorch/vision:v0.8.0', 'resnet18', pretrained=True)
img_enc.fc = torch.nn.Identity()

#################################
# generate vocabulary
#################################
data_path = "/mnt/disks/nvme3n1/chestXray/"


df = pd.read_csv(data_path+"uid_report_projection_label.csv")
word_2_id, id_2_word = generate_vocabulary(df)
vocab_size = len(word_2_id)
block_size = 512
print(id_2_word)


##################################
# generate train/validation sets
#################################
df = pd.read_csv(data_path+"uid_report_projection_label.csv")
df = df[df['filename'].notna()]
train_df, val_df = get_train_val_df(df)
train_dataset = chestXRayDataset(train_df, data_path, block_size, word_2_id, id_2_word)
train_dataloader = DataLoader(train_dataset, batch_size=32)
val_dataset = chestXRayDataset(val_df, data_path, block_size, word_2_id, id_2_word)
val_dataloader = DataLoader(val_dataset, batch_size=32)

print(f'There are {len(train_dataset) :,} samples for training, and {len(val_dataset) :,} samples for validation testing')

####################################
# create the encoder/decoder model
###################################

mconf = ImageEncoderReportDecoderConfig(vocab_size, block_size, n_embd=224)
model = ImageEncoderReportDecoder(mconf, img_enc)
model.load_state_dict(torch.load("./xray_model.pth"))


for x, y, z in val_dataset:
    x = x.unsqueeze(0)
    y = y.unsqueeze(0).unsqueeze(2)
    z = z.unsqueeze(0)
    ttt = sample(model, x, y[:,0,:], z = None, steps=30)
    zz = [id_2_word[k]  if  k != 2319 else '' for k in ttt[1:15]]
    print("predicted:", zz)
    y = y.tolist()[0]
    y = [item for sublist in y for item in sublist]
    yy = [id_2_word[k] if k != 2319 else '' for k in y[1:15]]
    print("actual:", yy)
