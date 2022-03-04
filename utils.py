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
def sample(model, x, y0, label, len_mask, steps):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    model.eval()
    generated = []
 
    #print("x >>>>>>>>>>:", x)
    y1 = y0
    reps = model.representation(x, label)
    #print(reps)

    for k in range(steps):
        
        #print("****************", y0)
        logits, _, pred = model.decode(reps, y0, None)
        #logits, _, pred = model(x, y0, None, label)
        
        #for j in range(len(y0[0])):
        #    y2 = torch.argmax(logits[:, j, :], dim=1)
        #    print("&&&&&&&&&&&&&&&&", y2)


        logits = logits[:, len(y0)-1, :]

        
        #logits[:, y1] = float('-inf')
        y1 = torch.argmax(logits,dim=1).unsqueeze(0)
        #print(y1.item())
        generated.append(y1.item())
        #print("y0 shape:", y0.shape)
        #print("y1 shape:", y1.shape)
        y0 = torch.cat((y0, y1), dim=1)
        
    return generated
