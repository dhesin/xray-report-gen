import numpy as np
import pandas as pd
import torchvision
import torch
import pickle
from dataset import chestXRayDataset
from torch.utils.data import Dataset,  DataLoader
import logging
from utils import set_seed
from model import ImageEncoderReportDecoder, ImageEncoderReportDecoderConfig
from trainer import Trainer, TrainerConfig
#import torchxrayvision as xrv
import argparse
from tokenizers import BertWordPieceTokenizer


logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

set_seed(42)
#torch.backends.cudnn.benchmark = False
#torch.use_deterministic_algorithms(True)

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

# DenseNet with all images; couldn't make it work
#img_enc_dense_all = xrv.models.DenseNet(weights="densenet121-res224-all")
#img_enc_dense_all.classifier = torch.nn.Identity()
#img_enc_dense_all.upsample = torch.nn.Identity()

# DenseNet with Chexpert images; couldn't make it work
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
img_enc_name = "ResNet18" # "UNet" # "ResNetAE" "Direct" "ResNet18"
img_enc = img_enc_resnet
img_enc_width, img_enc_height = img_enc.input_shape
img_enc_out_shape = img_enc.output_shape
block_size = img_enc_out_shape[0]


#for name, module in img_enc.named_modules():
#    print(name)


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
tokenizer = None
#vocab_size = tokenizer.get_vocab_size()

print("Vocab Size:", vocab_size)

##################################
# generate train/validation sets
#################################
train_dataset = chestXRayDataset(train_df, data_path, block_size, img_enc_width, img_enc_height, word_2_id, id_2_word, tokenizer)
val_dataset = chestXRayDataset(val_df, data_path, block_size, img_enc_width, img_enc_height, word_2_id, id_2_word, tokenizer)
print(f'There are {len(train_dataset) :,} samples for training, and {len(val_dataset) :,} samples for validation testing')


def parse_args():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--pretrained_encoder', type=str, default='./pretrained_encoder.pth')
    parser.add_argument('--train_decoder', action='store_true')
    parser.add_argument('--out_model_name', type=str, default='./xray_model_lr1-3.pth', help='output model name')
    parser.add_argument('--batch_size', type=int, default=16, help='')
    parser.add_argument('--embed_size', type=int, default=720, help='')

    args = parser.parse_args()
    return args


def main():
    # parse arguments
    args = parse_args()

    if args.pretrain:
        print("Pretraining with Contrastive Learning")



    ####################################
    # create the encoder/decoder model 
    ###################################
    torch.set_default_dtype(torch.float64)
    mconf = ImageEncoderReportDecoderConfig(vocab_size, block_size, args.embed_size, args.pretrain, \
            args.train_decoder, args.pretrained_encoder)
    model = ImageEncoderReportDecoder(mconf, img_enc, img_enc_out_shape, img_enc_name).float()
    
    if args.train_decoder:
        model.load_pretrained_encoder(args.pretrained_encoder)


    #print(model)

    #################################
    # set TrainerConfig and Trainer
    ###############################
    tokens_per_epoch = len(train_dataset) * block_size
    train_epochs = 300
    tconf = TrainerConfig(max_epochs=train_epochs, batch_size=args.batch_size, learning_rate=1.0e-3,
                          betas = (0.9, 0.95), weight_decay=0,
                          lr_decay=True, warmup_tokens=tokens_per_epoch, final_tokens=train_epochs*tokens_per_epoch,
                          ckpt_path=args.out_model_name,
                          num_workers=8,
                          pretrain = args.pretrain,
                          tokenizer= tokenizer)
    trainer = Trainer(model, train_dataset, val_dataset, tconf, word_2_id, id_2_word)
    trainer.train()



if __name__ == '__main__':
    main()   


