#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import  torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import shutil
import copy
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from statistics import mean
from collections  import OrderedDict
from collections  import namedtuple
import sys

class cache:
    
    def __init__(self, device):
        
        self.size = 10 ** 3
        self.W = torch.randn(self.size, device = device)
        self.A = torch.randn(self.size, device = device)
        self.time = torch.ones(self.size, device = device, dtype = torch.int32)
        self.epsilon = 1e-2
        self.hits = torch.tensor(0, device = device, dtype = torch.int64).view(1)
        self.misses = torch.tensor(0, device = device, dtype = torch.int64).view(1)
        self.device = device
        
def approx(cache, orig_W, orig_A):
    print('Called')
    
    s1 = orig_W.shape
    s2 = orig_A.shape

    orig_W = orig_W.flatten()
    orig_A = orig_A.flatten()
    if orig_W.shape != orig_A.shape:
        sys.exit("W-shape and A-shape unequal")

    # Remove indices where W = 0 or A = 0
    non_zero_mask = ~((orig_W == 0) | (orig_A == 0))
    W_ = orig_W[non_zero_mask]
    A_ = orig_A[non_zero_mask]

    W = W_
    A = A_

    first_time = True
    while(True):

        #print('\r Size is: {}, {}'.format(len(W), len(A)), end = '')

        # Find hits, misses
        if first_time:
            num_send = int(5 * (10 ** 7) / cache.size)
            num_elem = W.numel()
            dist = torch.zeros(W.shape, dtype = torch.float32, device = cache.device)
            LI_orig = torch.zeros(W.shape, dtype = torch.long, device = cache.device)
            for i in range(int(num_elem / num_send) + 1):
                start = i * num_send
                end = min((i + 1) * num_send, num_elem)
                if end == start:
                    break
                I = torch.cat((W[start:end].view(1,-1),A[start:end].view(1,-1)), dim = 0)
                S = torch.cat((cache.W.view(1,-1), cache.A.view(1,-1)), dim = 0)
                x = I.expand(S.shape[1],-1,-1)
                y = S.t().view(S.shape[1],2,1).expand(-1,-1,I.shape[1])
                dist[start:end], LI_orig[start:end] = torch.abs(x - y).sum(dim = 1).min(dim = 0)
            first_time = False
        else:
            # Calculate distances of the elements from newly added cache pair
            I = torch.cat((W.view(1,-1),A.view(1,-1)), dim = 0)
            S = buffer
            x = I.expand(S.shape[1],-1,-1)
            y = S.t().view(S.shape[1],2,1).expand(-1,-1,I.shape[1])
            dist_t = torch.abs(x - y).sum(dim = 1).squeeze(0)
#                 print('dist', dist)
#                 print('dist_t', dist_t)

            # Check if newly added cache pair better approximates some elements
            change = dist_t < dist
            change_t = ~ change
            LI_orig = change_t * LI_orig + change * buf_ind.expand_as(change)
            dist = change_t * dist + change * dist_t

#                 print('change', change)
#                 print('dist', dist)

            # Recalculation for elements that matched with removed cache pair
            incorrect = (LI_orig == buf_ind) & change_t
#                 print(incorrect)
            if incorrect.any():
                W_i = W[incorrect]
                A_i = A[incorrect]
                I = torch.cat((W_i.view(1,-1),A_i.view(1,-1)), dim = 0)
                S = torch.cat((cache.W.view(1,-1), cache.A.view(1,-1)), dim = 0)
                x = I.expand(S.shape[1],-1,-1)
                y = S.t().view(S.shape[1],2,1).expand(-1,-1,I.shape[1])
                dist[incorrect], LI_orig[incorrect] = torch.abs(x - y).sum(dim = 1).min(dim = 0)


        misses = ~(dist < cache.epsilon)
        t = misses.nonzero()

        if t.shape[0] != 0:
            max_lim = t[0]
        else:
            max_lim = W.shape[0]
        LI = LI_orig[:max_lim]

        if LI.shape[0] != 0: # Hits encountered

            # Update num hits
            cache.hits += max_lim

            # Update W, A corresponding to hits
            W[:max_lim] = cache.W[LI]
            A[:max_lim] = cache.A[LI]

            # Update time corresponding to hits

            size_lim = 5 * (10 ** 7)

            num_elem = LI.numel()
            used = torch.unique(LI)
            used_orig = used
            z = torch.zeros(used.numel(), dtype = torch.long, device = cache.device)
            curr_inds = torch.arange(start = 0, end = used.numel(), device = cache.device)

            start = int(num_elem - size_lim / used.numel())
            start = max(start, 0)
            end = num_elem
            adder = 0

            # LI processed in a batchwise fashion, starting from the end to the start
            # Batch size is dynamic, so as to keep the number of elements in "LI[start:end] == used" constant, as size of 
            # used decreases throughout the while loop iterations
            while(True):

                if end == start:
                    break

                x = LI[start:end].view(-1,1) == used.view(1,-1)   
                x = x.type(torch.int32)
                x = x * torch.arange(start = 1, end = x.shape[0]+1, device = cache.device).view(-1,1).expand_as(x)
                z[curr_inds] = x.shape[0] - 1 - torch.argmax(x, dim = 0) + adder

                rem_bool = torch.sum(x, dim = 0) == 0
                curr_inds = curr_inds[rem_bool]
                used = used[rem_bool]

                if (used.numel() == 0):
                    break

                adder += end - start
                end = start
                start = int(start - size_lim / used.numel())
                start = max(start, 0)

            cache.time += LI.shape[0]
            cache.time[used_orig] = 0
            cache.time[used_orig] += z

#                 print('Time', cache.time)
#                 print('W', W)
#                 print('A', A)

        if t.shape[0] != 0: # A miss encountered

            cache.misses += 1

            # Update stored values in cache
            ind = torch.argmax(cache.time)
            cache.W[ind] = W[max_lim]
            cache.A[ind] = A[max_lim]
            cache.time += 1
            cache.time[ind] = 0

#                 print('Miss at', max_lim, W[max_lim], A[max_lim])
#                 print('New cache W', cachcache
#                 print('New cache A', cache.A)
#                 print('New time', cache.time)

            if max_lim == W.shape[0] - 1:
                break

            # Update W, A, for next cycle
            buffer = torch.cat((W[max_lim].view(1,-1),A[max_lim].view(1,-1)), dim = 0)
            buf_ind = ind
            W = W[max_lim + 1:]
            A = A[max_lim + 1:]
            dist = dist[max_lim + 1:]
            LI_orig = LI_orig[max_lim + 1:]

#                 print('W', W)
#                 print('A', A)
#                 print()

        else:
            break

    orig_W[non_zero_mask] = W_
    orig_A[non_zero_mask] = A_

    hits = cache.hits.item()
    misses = cache.misses.item()

    #cache.hits = torch.tensor(0, device = cache.device, dtype = torch.int64).view(1)
    #cache.misses = torch.tensor(0, device = cache.device, dtype = torch.int64).view(1)
    cache.hits -= hits
    cache.misses -= misses
    
    return orig_W.view(s1), orig_A.view(s2), hits, misses
            

