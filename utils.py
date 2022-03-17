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
def sample(model, x, y0, label, len_mask, steps, word_2_id):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    model.eval()
    generated = []
 
    y1 = y0
    y2 = y0
    reps = model.representation(x, label)
    #print(reps)

    for k in range(steps):
        
        logits, _, pred = model.decode(reps, y0, None, label)

        #logits = logits[:,0,:]
        #print("logits shape:", logits.shape)
        #print("input shape:", y0.shape)
        assert(logits.shape[1] == y0.shape[1])
        logits = logits[:,-1,:]

        
        #logits[:, y1] = float('-inf')
        #logits[:, y2] = float('-inf')


        y2 = y1
        y1 = torch.argmax(logits,dim=1).unsqueeze(0)
        generated.append(y1.item())

        #if y1.item() == word_2_id['<eos>']:
        #    break;

        #print("y0 shape:", y0.shape)
        #print("y1 shape:", y1.shape)
        y0 = torch.cat((y0, y1), dim=1)
        
    return generated
