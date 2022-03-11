import math
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Transformer
from torch import linalg as LA

logger = logging.getLogger(__name__)


class ImageEncoderReportDecoderConfig:

    def __init__(self, vocab_size, block_size, n_embd, pretrain, train_decoder, pretrained_encoder_model):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.embd_pdrop = 0.3
        self.n_lbl_class = 4
        self.n_lbls = 14
        self.tgt_drop_p = 0.3
        self.pretrain = pretrain
        self.train_decoder = train_decoder
        self.pretrained_encoder_model = pretrained_encoder_model,

class ImageEncoderReportDecoder(nn.Module):

    def __init__(self, config, img_enc, img_enc_out_shape, img_enc_name="ResNet18"):
        super().__init__()

        self.cnf = config
        self.img_enc_name = img_enc_name
        
        self.label_emb = nn.Embedding(config.n_lbl_class, config.n_embd)
        self.pos_emb_labels = nn.Parameter(torch.zeros(1, config.n_lbls, config.n_embd))

        self.img_enc = img_enc
        self.img_enc_linear = nn.Linear(img_enc_out_shape[1], config.n_embd)
        
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        encoder_layer = nn.TransformerEncoderLayer(d_model=config.n_embd, nhead=36, dim_feedforward=2048, \
                dropout=0.3, activation='gelu', batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)


        self.contrastive_head = nn.Linear(config.n_embd, 1)

        self.tgt_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.tgt_drop = nn.Dropout(config.tgt_drop_p)

        decoder_layer = nn.TransformerDecoderLayer(d_model=config.n_embd, nhead=36, dim_feedforward=2048, \
            dropout=0.3, activation='gelu', batch_first=True, norm_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        self.dcd_proj = nn.Linear(config.n_embd, config.vocab_size)
        
        self.block_size = config.block_size
        self.apply(self._init_weights)

        self.register_buffer("mask", torch.triu(torch.ones(config.block_size, config.block_size)*float('-inf'),diagonal=1))
        self.mask = self.mask < 0
        #print(self.mask)


        #logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, targets, len_mask, labels):

        loss = None
        logit = None
        prediction = None

        
        reps, loss = self.representation(x, labels)
        logit, loss, prediction = self.decode(reps, targets, len_mask, labels)

        return logit, loss, prediction

    def configure_optimizers(self, train_config):
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def get_block_size(self):
        return self.block_size

    def representation(self, x, labels):

        x_org = x
        if labels != None:
            label_emb = self.label_emb(labels)
            label_emb = label_emb + self.pos_emb_labels

        if self.img_enc_name == "ResNet18" or self.img_enc_name == "UNet":
            x = torch.cat((x.unsqueeze(1),x.unsqueeze(1),x.unsqueeze(1)), dim=1)
            x_org = x_org.unsqueeze(1)
        elif self.img_enc_name == "ResNetAE":
            x = x.unsqueeze(1)

        with torch.no_grad():
            x = self.img_enc(x)
            #print("img enc output shape:", x['z'].shape, "   ", x['out'].shape)
            if self.img_enc_name == "ResNetAE":
                b,t,e1,e2 = x['z'].shape
                x = x['z'].reshape((b,t,-1))
            elif self.img_enc_name == "UNet":
                pass
                #x = x_org + x
            #print("img enc shape:", x.shape)

        if len(x.shape) < 3:
            x = x.unsqueeze(2)
        elif len(x.shape) > 3:
            x = x.squeeze(1)

        x = self.img_enc_linear(x)
        b, t, e = x.size()
        x = x + self.pos_emb[:, :t,:]
        x = self.drop(x)
        
        x = torch.cat((x, label_emb), dim=1)

        reps = self.transformer_encoder(x)

        cont_loss = None
        #if self.cnf.pretrain:
        #    cont_loss = self.contrastive(reps, labels)
        
        #print("cont loss:", cont_loss)
        #print("reps shape:", reps)
        return reps, cont_loss

    def decode(self, reps, targets, len_mask, labels):

        tgt_emb = self.tgt_emb(targets)
        bt, tt, et = tgt_emb.size()
        tgt_emb = tgt_emb + self.pos_emb[:, :tt, :]
        tgt_emb = self.tgt_drop(tgt_emb)
        
        if len_mask != None:
            len_mask = len_mask > 0
            dec_out = self.transformer_decoder(tgt_emb[:,:-1,:], reps, tgt_mask=self.mask[:-1,:-1], tgt_key_padding_mask=len_mask[:,:-1])
            #dec_out = self.transformer_decoder_2(tgt_emb[:,:-1,:], dec_out, tgt_mask=self.mask[:-1,:-1], tgt_key_padding_mask=len_mask[:,:-1])
        else:
            dec_out = self.transformer_decoder(tgt_emb, reps)
            #dec_out = self.transformer_decoder_2(tgt_emb, dec_out)

        logits = self.dcd_proj(dec_out)

        loss = None
        if len_mask != None:
            loss = F.cross_entropy(logits.view(-1,logits.shape[-1]), targets[:,1:].reshape(-1), reduction='none')

            #logits_weights = torch.where(len_mask[:,1:] == True, 0.4, 0.6).view(-1)
            #loss = loss * logits_weights
            #loss = loss.mean()

            #cont_loss = self.contrastive(dec_out, labels)
            #loss = cont_loss + loss

        prediction = torch.argmax(logits,dim=-1)
        return logits, loss, prediction

    def contrastive(self, reps, labels):

        one_hot_labels = torch.vstack([F.one_hot(labels[i].long(), 4).unsqueeze(0) for i in range(labels.shape[0])])
        #print(one_hot_labels.shape)
        one_hot_labels = torch.cat((one_hot_labels[:,:,:1],one_hot_labels[:,:,2:3]), dim=2).contiguous().to('cuda')

        #one_hot_labels = one_hot_labels[:, :, :3].contiguous().to('cuda')
        #print(one_hot_labels)
        
        label_dot = torch.full((labels.shape[0], labels.shape[0]), float('-inf'))
        num_com_labels = torch.zeros(labels.shape[0], labels.shape[0])
        for i in range(labels.shape[0]):
            for j in range(i+1, labels.shape[0]):
                num_labels  = (one_hot_labels[i] + one_hot_labels[j]).sum(dim=1)
                num_labels = num_labels > 1
                num_labels = num_labels.sum()
                if num_labels != 0:
                    label_dot[i,j] = torch.div((one_hot_labels[i] * one_hot_labels[j]).sum(),num_labels)
                    torch._assert(label_dot[i,j] >=0 and label_dot[i,j] <= 1.0, label_dot)
                    num_com_labels[i,j] = num_labels


        reps = self.contrastive_head(reps)
        reps_norm = LA.vector_norm(reps, ord=2, dim=1)

        diff = 0
        num_diff = 0
        for i in range(labels.shape[0]):
            for j in range(i+1, labels.shape[0]):
                if label_dot[i,j] > float('-inf'):
                    reps_dot = torch.div((reps[i] * reps[j]).sum(), reps_norm[i]*reps_norm[j])
                    reps_dot = (reps_dot + 1.0)/2.0

                    torch._assert(reps_dot >=0 and reps_dot <= 1.0, reps_dot)
                    diff = diff - label_dot[i,j]*torch.log(reps_dot) - (1-label_dot[i,j])*torch.log(1-reps_dot)
                    num_diff = num_diff + 1
        
        loss = torch.Tensor([0.0])
        if num_diff != 0:
            loss = diff/num_diff
        else:
            print("Nothing to contrast in batch")

        return loss.to('cuda')


    def load_pretrained_encoder(self, model_name):

        state_dict = torch.load(model_name)

        self.pos_emb_labels = torch.nn.Parameter(state_dict['pos_emb_labels'])
        self.label_emb.load_state_dict(state_dict, strict=False)
        self.pos_emb = torch.nn.Parameter(state_dict['pos_emb'])
        #encoder_layer.load_state_dict(state_dict, strict=False)
        self.transformer_encoder.load_state_dict(state_dict, strict=False)

