import torch
from torch import nn, optim
from time import sleep
import numpy as np
import gc
import csv
import sys
import datetime

from iqtests import IQ_001_Counting as iq1
from iqtests import IQ_002_BumpTest as iq2
from iqtests import IQ_003_ShortestPath as iq3

from Models.torchRNN import train_model, RegisteredGRU

cursus = [(iq2.problem(i, sequential=True)[0], f"IQ2 ({i})") for i in range(2, 10)] + \
         [(iq3.problem(i, sequential=True)[0], f"IQ3 ({i})") for i in range(2, 10)]

vocab = iq2.problem(10, sequential=True)[1]
model = RegisteredGRU(vocab.vector_size, 128, 10).cuda()
print(vocab.vector_size)

epochs = 150000
sequential = True
fail_threshold = 0.9
log_freq = 100

results = list()

filename_prefix = "model_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
for i, (exp, name) in enumerate(cursus):
    print(f"Step #{i+1} : {name}")
    opt = optim.AdamW(model.parameters())
    filename = filename_prefix + f"_step{i}.bin"
    opt_filename = filename_prefix + f"_opt_step{i}.bin"
    loss, accuracy, end_epoch = train_model(model, epochs, exp, vocab,
                                            opt,
                                            nn.MSELoss(), batch_size=32,
                                            loss_threshold=-0.00001,
                                            acc_threshold=0.99,
                                            log_freq=log_freq)
    torch.save(model.state_dict(), filename)
    torch.save(opt.state_dict(), opt_filename)
