import numpy as np
import pandas as pd
import torchvision
import torch
from dataset import chestXRayDataset
import pickle
from torch.utils.data import Dataset,  DataLoader
import logging
from utils import set_seed, sample
from mymodel import ImageEncoderReportDecoder, ImageEncoderReportDecoderConfig
from trainer import Trainer, TrainerConfig
from torchtext.data.metrics import bleu_score
from tqdm import tqdm
#import torchxrayvision as xrv
from tokenizers import BertWordPieceTokenizer
import re

logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

set_seed(42)


##################################
# load pretrained nets for image encoding
##################################
# Resnet 18  img_enc_width = img_enc_height = 224  / img_enc_out_shape = (512,1) / block_size = 512 / rgb = True
img_enc_resnet = torch.hub.load('pytorch/vision:v0.8.0', 'resnet18', pretrained=True)
img_enc_resnet.fc = torch.nn.Identity()
img_enc_resnet.input_shape = (224, 224)
img_enc_resnet.output_shape = (512, 1)


# Unet img_enc_width = img_enc_height = 256  / img_enc_out_shape = (256, 256) / block_size = 256 / rgb = True
img_enc_unet = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=True)
img_enc_unet.input_shape = (256, 256)
img_enc_unet.output_shape = (256, 256)

# DenseNet with all images; couldn't make it work !!!!!!!!!!!!
#img_enc_dense_all = xrv.models.DenseNet(weights="densenet121-res224-all")
#img_enc_dense_all.classifier = torch.nn.Identity()
#img_enc_dense_all.upsample = torch.nn.Identity()

# DenseNet with Chexpert images; couldn't make it work !!!!!!!!!!!!!!!!!!!
#img_enc_dense_chex = xrv.models.DenseNet(weights="densenet121-res224-chex")
#img_enc_dense_chex.classifier = torch.nn.Identity()
#img_enc_dense_chex.upsample = torch.nn.Identity()

# ResNet AutoEncoder
#img_enc_ae = xrv.autoencoders.ResNetAE(weights="101-elastic")
#img_enc_ae.input_shape = (224, 224)
#img_enc_ae.output_shape = (512, 9)

# Directly feed the image into transformer encoder
img_enc_direct = torch.nn.Identity()
img_enc_direct.input_shape = (224, 224)
img_enc_direct.output_shape = (224, 224)


# Select one of the above
img_enc_name = "UNet" # "UNet" # "ResNetAE" "Direct"
img_enc = img_enc_unet
img_enc_width, img_enc_height = img_enc.input_shape
img_enc_out_shape = img_enc.output_shape
block_size = img_enc_out_shape[0]


for name, module in img_enc.named_modules():
    print(name)

#################################
# load vocabulary and dataframes
#################################
data_path = "/home/desin/CS224N/data/chestXray/"

with open("./db_vocab.pkl", "rb") as cache:
    db_vocab = pickle.load(cache)
    word_2_id = db_vocab["word_2_id"]
    id_2_word = db_vocab["id_2_word"]

vocab_size = len(word_2_id)
assert(len(id_2_word) == len(word_2_id))


with open("./db_datasets.pkl", "rb") as cache:
    db_database = pickle.load(cache)
    train_df = db_database["train_df"]
    val_df = db_database["val_df"]



#tokenizer = BertWordPieceTokenizer("bert-vocab.txt")
#print(tokenizer)
#vocab_size = tokenizer.get_vocab_size()
#print("vocabulary size:", vocab_size)
tokenizer = None

##################################
# generate train/validation sets
#################################
train_dataset = chestXRayDataset(train_df, data_path, block_size, img_enc_width, img_enc_height, word_2_id, id_2_word, tokenizer)
val_dataset = chestXRayDataset(val_df, data_path, block_size, img_enc_width, img_enc_height, word_2_id, id_2_word, tokenizer)
print(f'There are {len(train_dataset) :,} samples for training, and {len(val_dataset) :,} samples for validation testing')

####################################
# create the encoder/decoder model
###################################

torch.set_default_dtype(torch.float64)
mconf = ImageEncoderReportDecoderConfig(vocab_size, block_size, n_embd=720, pretrain=False, train_decoder=False, pretrained_encoder_model=None)
model = ImageEncoderReportDecoder(mconf, img_enc, img_enc_out_shape, img_enc_name).float()
model.load_state_dict(torch.load("./xray_model_lr3-3.pth"))
#model.img_enc = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=True)



tgts = []
preds = []


print("len val dataset:", len(val_dataset))
pbar = tqdm(enumerate(val_dataset)) 

for it, (x, y, _, label) in pbar:

    if it >= len(val_dataset)-1:
        break
    #print("decoder input shape:", x)
    x = x.unsqueeze(0)
    y = y.unsqueeze(0).unsqueeze(2)
    gen = sample(model, x, y[:,0,:], label, None, steps=30, word_2_id=word_2_id)

    y = [item.item() for sublist in y[0] for item in sublist]
    tgts.append(torch.IntTensor(y))
    preds.append(torch.IntTensor(gen))
    #print(y)
    #print(ttt)

    gen_text = [id_2_word[k]  if  k != word_2_id['<eos>'] else '' for k in gen]
    #zz = [tokenizer.id_to_token(k)  if  k != 3 else '' for k in gen]
    print("predicted:", gen_text)
    y_text = [id_2_word[k]  if  k != word_2_id['<eos>'] else '' for k in y]
    #print("actual:", y_text)
    pbar.set_description(f"{it}/{len(val_dataset)}")



tgts = torch.vstack(tgts).cpu().numpy().tolist()
preds = torch.vstack(preds).cpu().numpy().tolist()
tgts_list = []
preds_list = []
for i in range(len(tgts)):
    try:
        eos_ind = tgts[i].index(word_2_id['<eos>'])
    except:
        eos_ind = len(tgts[i])-1

    tgts_list.append(tgts[i][:eos_ind])
    #tgts_list[-1] = [tokenizer.decode(tgts_list[-1]).split()]
    tgts_list[-1] = [[id_2_word[x] for x in tgts_list[-1]]]

for i in range(len(preds)):
    try:
        eos_ind = preds[i].index(word_2_id['<eos>'])
    except:
        eos_ind = len(preds[i])-1
    
    preds_list.append(preds[i][:eos_ind])
    #preds_list[-1] = tokenizer.decode(preds_list[-1]).split()
    preds_list[-1] = [str(id_2_word[x]) for x in preds_list[-1]]

assert(len(preds_list) == len(tgts_list))
#print(preds_list)
#print(tgts_list)
test_bleu_1 = bleu_score(preds_list, tgts_list, max_n=1, weights=[1])   
test_bleu_2 = bleu_score(preds_list, tgts_list, max_n=2, weights=[0,1])
test_bleu_3 = bleu_score(preds_list, tgts_list, max_n=3, weights=[0,0,1])
test_bleu_4 = bleu_score(preds_list, tgts_list, max_n=4, weights=[0,0,0,1])
test_bleu_5 = bleu_score(preds_list, tgts_list, max_n=5, weights=[0,0,0,0,1])
print(f"Bleu Scores:1:{test_bleu_1} \t 2:{test_bleu_2} \t 3:{test_bleu_3} \t 4:{test_bleu_4} \t 5:{test_bleu_5}")
