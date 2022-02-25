import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def sample(model, x, y0, z, steps):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    model.eval()
    generated = []
    for k in range(steps):
        
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        logits, _, pred = model(x_cond, y0, z)
        logits = logits[:, y0.shape[1]-1, :]
        y1 = torch.argmax(logits,dim=1).unsqueeze(0)
        #print(y1.item())
        generated.append(y1.item())
        #print("y0 shape:", y0.shape)
        #print("y1 shape:", y1.shape)
        y0 = torch.cat((y0, y1), dim=1)
        
    return generated
