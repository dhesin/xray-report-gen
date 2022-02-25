import pandas as pd
import numpy as np
import re
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
import torch

TRAIN_SIZE = 0.8
SEQUENCE_LEN = 1500

def generate_vocabulary(df):

    #data_path = "/mnt/disks/nvme3n1/chestXray/"
    #df = pd.read_csv(data_path+"uid_report_projection_label.csv")
    df = df.fillna('')
    
    words = []
    for (report, finding) in zip(df.impression, df.findings):
        rp = re.findall(r"[\w']+|[.,!?;]", report)
        fg = re.findall(r"[\w']+|[.,!?;]", finding)
        words.append(rp)
        words.append(fg)
        #words.append(str.split(report))

    vocab = [x for sublist in words for x in sublist]
    #for item in vocab:
    #    print(item)

    vocab = sorted(np.unique(vocab))

    word_2_id = {}
    id_2_word = {}
    for ind, word in enumerate(vocab):
        word_2_id[str(word)] = ind
        id_2_word[ind] = word

    vocab_size = len(vocab)
    word_2_id['<eos>'] = vocab_size
    id_2_word[vocab_size] = '<eos>'
    word_2_id['<start>'] = vocab_size+1
    id_2_word[vocab_size+1] = '<start>'

    #print(word_2_id)
    #print(id_2_word)
    print("len vocab:", len(vocab))
    return word_2_id, id_2_word

#word_2_id, id_2_word = generate_vocabulary()


def get_train_val_df(df):
  """
    Separates the dataframe into training and validation sets. Splits by subject id.
  """
  train_split = TRAIN_SIZE
  ids = df.uid.unique()
  np.random.seed(1)
  train_uids = np.random.choice(ids, size = int(len(ids)*train_split), replace = False)

  df['in_train'] = None
  df['in_train'] = df["uid"].apply(lambda x: x in train_uids)
  train_df = df[df['in_train'] == True]
  val_df = df[df['in_train'] == False]

  return train_df, val_df

def one_hot_row(row, size):
    
    one_hot_row = []
    for a in row:
        b = np.zeros(size)
        b[int(a)] = 1
        one_hot_row.append(b)
 
    flat_list = np.vstack(one_hot_row).flatten()
    return flat_list
    

class chestXRayDataset(Dataset):
    def __init__(self, df, img_dir, block_size, word_2_id, id_2_word):
        
        self.block_size = block_size

        df = df.reset_index()
        self.img_labels = df[['No Finding','Enlarged Cardiomediastinum','Cardiomegaly','Lung Lesion', \
                              'Lung Opacity','Edema','Consolidation','Pneumonia','Atelectasis',\
                              'Pneumothorax','Pleural Effusion','Pleural Other','Fracture','Support Devices']]
        self.word_2_id = word_2_id
        self.id_2_word = id_2_word

        self.img_labels = self.img_labels.fillna(2)
        df.impression = df.impression.fillna('')
        df.findings = df.findings.fillna('')


        self.img_labels = self.img_labels + 1
        self.num_labels = len(self.img_labels.columns)
        self.one_hot_labels = self.img_labels.apply(lambda x: one_hot_row(x,4), axis=1)
        self.one_hot_labels = np.vstack(self.one_hot_labels)
        self.img_labels = self.img_labels.to_numpy()

        self.findings = df['findings'].apply(lambda x: re.findall(r"[\w']+|[.,!?;]", x))
        self.findings = self.findings.apply(lambda x: [self.word_2_id[str(word)] for word in x])
        self.reports = df['impression'].apply(lambda x: re.findall(r"[\w']+|[.,!?;]", x))
        self.reports = self.reports.apply(lambda x: [self.word_2_id[str(word)] for word in x])
        self.rep_lens = self.reports.apply(lambda x: len(x)+2)
        self.reports = self.reports.apply(lambda x: np.pad(x, (1,block_size-1-len(x)), constant_values=(word_2_id['<start>'],word_2_id['<eos>'])))
        
        self.img_files = df['filename'].apply(lambda x: os.path.join(img_dir+"/images/images_normalized/", x))

        #mean, std = self.__get_mean_std()
        #print("mean:", mean)
        #print("std:", std)

    def __get_mean_std(self):
        
        means = []
        stds = []
        for file in self.img_files:
            image = Image.open(file)
            image = np.asarray(image).astype(float)/255.0
            means.append(np.mean(image))
            stds.append(np.std(image))
            
        mean = np.mean(means)
        std = np.std(stds)
        
        return mean, std

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = Image.open(self.img_files.iloc[idx])
        image = image.resize((224,224), resample=Image.BICUBIC)
        image = np.asarray(image).astype(float)/255.0  

        #tmp = np.stack((image,image,image),0)
        tmp = image
        mean = np.array([0.624, 0.624, 0.624])
        std = np.array([0.038, 0.038, 0.038])
        tmp[0] = (tmp[0] - mean[0]) / std[0]
        tmp[1] = (tmp[1] - mean[1]) / std[1]
        tmp[2] = (tmp[2] - mean[2]) / std[2]
        image = tmp
        
        
        one_hot = self.one_hot_labels[idx]
        label = self.img_labels[idx]
        report = self.reports[idx]
        len_mask = [0 if i < self.rep_lens[idx] else 1 for i in range(self.block_size)]
        """rn = np.random.uniform()
        if rn < 0.10:
            image = np.flip(image, 1)
        elif rn >= 0.10 and rn < 0.20:
            image = np.flip(image, 2)
        elif rn >= 0.20 and rn < 0.30:
            image = np.flip(image, (1,2))
        elif rn >= 0.30 and rn < 0.40:
            image = np.transpose(image, (0,1,2))
        elif rn >= 0.50 and rn < 0.60:
            rn2 = np.random.uniform(20,60)
            image = np.roll(image, int(rn2), 0)
        elif rn >= 0.60 and rn < 0.70:
            rn2 = np.random.uniform(20,60)
            image = np.roll(image, int(rn2), 1)
        elif rn >= 0.70 and rn < 0.80:
            image = np.flip(image, (2,1))"""

        return torch.FloatTensor(image.astype(np.float32)), torch.LongTensor(report), torch.ByteTensor(len_mask)
    
    def collate_fn(self, samples):
        images, reports, len_masks = [], [], []
        for image, report, len_mask in samples:
            images.append(image)
            reports.append(report)
            len_masks.append(len_mask)

        
        #print("labels:", labels)
        reports = pad_sequence(reports, batch_first=True, padding_value=self.word_2_id['<eos>'])
        images = pad_sequence(images, batch_first=True)
        len_masks = torch.vstack(len_masks)
        #print(reports.shape)
        #print(rep_lens)
        return images, reports, len_masks


#data_path = "/tf/data/chestXray/"
#df = pd.read_csv(data_path+"uid_report_projection_label.csv")
#dataset = chestXRayDataset(df, data_path, word_2_id, id_2_word)
