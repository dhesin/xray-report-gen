import math
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Transformer

logger = logging.getLogger(__name__)


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook




class ImageEncoderReportDecoderConfig:

    def __init__(self, vocab_size, block_size, n_embd):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.embd_pdrop = 0.1


class ImageEncoderReportDecoder(nn.Module):


    def __init__(self, config, img_enc):
        super().__init__()

        self.cnf = config
        self.img_enc = img_enc

        #self.img_enc.layer4.register_forward_hook(get_activation('resnet_layer4'))


        self.img_enc_linear = nn.Linear(1, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        self.tgt_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.enc_dcd = nn.Transformer(d_model=config.n_embd, nhead=8, num_encoder_layers=6, num_decoder_layers=6, \
                dim_feedforward=2048, dropout=0.1, activation='relu', custom_encoder=None, batch_first=True, \
                custom_decoder=None)
        self.decoder = nn.Linear(config.n_embd, config.vocab_size)
        self.block_size = config.block_size
        self.apply(self._init_weights)

        self.register_buffer("mask", torch.triu(torch.ones(config.block_size, config.block_size)))
        print(self.mask)


        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

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

    def forward(self, x, targets, len_mask):

        x = torch.cat((x.unsqueeze(1),x.unsqueeze(1),x.unsqueeze(1)), dim=1)
        x = self.img_enc(x)

        #print(activation['resnet_layer4'].shape)

        x = x.unsqueeze(2)
        x = self.img_enc_linear(x)
        b, t, e = x.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        
        x = self.drop(x + self.pos_emb[:, :t, :])
        #print("x shape:", x.shape)
        #print("target shape:", targets.shape)
        tgt_emb = self.tgt_emb(targets)
        bt, tt, et = tgt_emb.size()
        #print("tgt_emb shape:", tgt_emb.shape)
        #print("mask shape:", self.mask.shape)
        tgt_emb = self.drop(tgt_emb + self.pos_emb[:, :tt, :])
        #print("tgt shape:", tt)

        logits = self.enc_dcd(x, tgt_emb, tgt_mask=self.mask[:tt,:tt], tgt_key_padding_mask=len_mask)
        logits = self.decoder(x)
        logits = logits[:,:tt,:].contiguous()


        #print(len_targets)
        #print(targets)
        #log_cat = []
        #tgt_cat = []
        #for i in range(logits.shape[0]):
        #    log_cat.append(logits[i,:len_targets[i],:])
        #    tgt_cat.append(targets[i,:len_targets[i]])
    
        #log_cat = torch.cat(log_cat)
        #tgt_cat = torch.cat(tgt_cat)
        #print(log_cat.shape)
        #print(tgt_cat.shape)
        #loss = F.cross_entropy(log_cat, tgt_cat)


        #print("logits shape:", logits.shape)
        #print("targets shape:", targets.shape)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1,logits.shape[-1]), targets.view(-1))

            if len_mask is not None:
                loss = loss.masked_fill(len_mask.view(-1) == 1, 0)
        
        prediction = torch.argmax(logits,dim=-1)
        return logits, loss, prediction

    def configure_optimizers(self, train_config):
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def get_block_size(self):
        return self.block_size

