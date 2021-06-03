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
from Models.torchRNN import LSTMBasic
from Models.torchTransformer import EncoderStack
from ivlib.utils import pprint

models = problems = list()
if "single" in sys.argv:
    models = [
        ["RegisteredGRU(128) Reg(3)", lambda vs: RegisteredGRU(vs, 128, 3)],
    ]
    problems = [(iq2, (2,))]
else:
    # This is the problems and models supposed to converge
    models = [
        ["RegisteredGRU(128) Reg(3)", lambda vs: RegisteredGRU(vs, 128, 3)],
        # ["LSTM (16)", lambda vs: LSTMBasic(vs, 16)],
        # ["LSTM (64)", lambda vs: LSTMBasic(vs, 64)],
        # ["LSTM (256)", lambda vs: LSTMBasic(vs, 256)],
        # #["Encoder (32)", lambda vs: EncoderStack(32, nheads=8)],
        # #["Encoder (64)", lambda vs: EncoderStack(64, nheads=8)],
        # ["LSTM(16) Stacked(4)", lambda vs: LSTMBasic(vs, 16, 4)],
        # ["GRU (16)", lambda vs: LSTMBasic(vs, 16)],
        # ["GRU (64)", lambda vs: LSTMBasic(vs, 64)],
        # ["GRU (256)", lambda vs: LSTMBasic(vs, 256)],
    ]

    problems = [
        (iq1, (10,)),
        (iq2, (2, 3, 4, 5, 6, 7, 8)),
        (iq3, (2, 3, 4, 5, 6, 7, 8)),
    ]

epochs = 150000
sequential = True
fail_threshold = 0.9
log_freq = 20

results = list()

# iq3 size 8 gives a vsize of 73
vocab_max_size = iq3.problem(8)[1].vector_size

# That mess about the vocab to choose is icky. Let's have a hard thinking about cross-problem
# vocabs soon.
for model_name, model_init in models:
    print(model_name, "cursus training")
    model = model_init(vocab_max_size).cuda()
    opt = optim.AdamW(model.parameters())
    for problem_module, problem_sizes in problems:
        vocab = problem_module.problem(problem_sizes[-1], sequential=sequential)[1]
        vocab.vector_size = vocab_max_size
        for problem_size in problem_sizes:
            print(problem_module.__name__, problem_size)
            G, _ = problem_module.problem(problem_size, sequential=sequential)
            loss, accuracy, end_epoch = train_model(model, epochs, G, vocab,
                                         opt,
                                         nn.MSELoss(), batch_size=32,
                                         loss_threshold=-0.00001,
                                         acc_threshold=0.99,
                                         log_freq=log_freq)
            results.append([model_name, problem_module.__name__, problem_size, end_epoch, loss, accuracy])
            if accuracy < fail_threshold:
                break

if not "single" in sys.argv:
    writer = csv.writer(open("benchmark-"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+".csv", 'w'))
    header = f"# Benchmark on {epochs} epochs"
    if sequential:
        header += ", sequential"
    writer.writerow([header])
    writer.writerow(["model_name", "problem_name", "problem_size", "end_epoch", "loss", "accuracy"])
    for row in results:
            writer.writerow(row)
pprint(results)
