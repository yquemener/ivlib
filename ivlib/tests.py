import sys
import random
import numpy as np
from inspect import getmembers, isfunction

from iqtests import IQ_001_Counting as iq1
from iqtests import IQ_002_BumpTest as iq2
from iqtests import IQ_003_ShortestPath as iq3


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def test_iq1_1():
    G, V = iq1.problem(10, False)
    s = G.generate_batch(1)
    expected = ([[['box', 'contains', 'trinket'], ['box', 'contains', 'trinket'], ['noise2', 'noise1', 'noise1'],
       ['box', 'contains', 'trinket'], ['box', 'contains', 'trinket'], ['box', 'contains', 'trinket'],
       ['noise1', 'box', 'trinket'], ['box', 'contains', 'trinket'], ['box', 'contains', 'trinket'],
       ['trinket', 'noise1', 'trinket'], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]], [0.7], [10])
    if str(s) == str(expected):
        return True
    else:
        return False


def test_iq1_sequential():
    G, V = iq1.problem(10, True)
    prob, sol, length = G.generate_batch(2)
    oh = V.one_hot(prob)
    if oh.shape == (2, 60, 27):
        return True
    else:
        return False


def test_iq2():
    G, V = iq2.problem(2, False)
    s = G.generate_batch(1)
    expected = ([[['problem', 'is', '0'], ['start', 'is', 'room_1_0'], ['destination', 'is', 'room_1_1'], ['direction', 'is', 'room_0_0'], ['room_0_0', 'path', 'room_1_0'], ['room_1_0', 'path', 'room_0_0'], ['room_1_0', 'path', 'room_1_1'], ['room_1_1', 'path', 'room_1_0'], ['0', '0', '0']]], [1.0])
    if str(s) == str(expected):
        return True
    else:
        return False


def test_iq2_sequential():
    G, V = iq2.problem(2, True)
    s = G.generate_batch(2)[0]
    if V.one_hot(s).shape == (2, 36, 13):
        return True
    else:
        return False


def test_iq3_1():
    G, V = iq3.problem(2, False)
    s = G.generate_batch(1)
    expected = ([[['problem', 'is', '0'], ['start', 'is', 'room_1_0'], ['destination', 'is', 'room_1_1'], ['direction', 'is', 'room_0_0'], ['room_0_0', 'path', 'room_1_0'], ['room_1_0', 'path', 'room_0_0'], ['room_1_0', 'path', 'room_1_1'], ['room_1_1', 'path', 'room_1_0']]], [0.0])
    if str(s) == str(expected):
        return True
    else:
        return False


def test_iq3_sequential():
    G, V = iq3.problem(2, True)
    s = G.generate_batch(2)[0]
    expected = (2, 32, 13)
    if V.one_hot(s).shape == expected:
        return True
    else:
        return False


def test_solve_iq2():
    G, V = iq2.problem(2, False)
    prob, sol = G.generate_batch(100)
    start = dir = None
    for i,p in enumerate(prob):
        out = 0.0
        for src,arc,dst in p:
            if src=="start" and arc=="is":
                start = dst
            if src=="direction" and arc=="is":
                dir = dst
            if src==start and arc=="path" and dst==dir:
                out=1.0
                break
        if out!=sol[i]:
            return False
    return True


def test_solve_iq3():
    G, V = iq2.problem(2, False)
    prob, sol = G.generate_batch(100)
    start = dir = None
    for i,p in enumerate(prob):
        return False
    return True


def run_all():
    functions_list = [o for o in getmembers(sys.modules[__name__]) if isfunction(o[1]) and o[0].startswith("test_")]
    for f in functions_list:
        random.seed(0)
        np.random.seed(0)
        res = f[1]()
        color = None
        if res:
            color = bcolors.OKGREEN
        else:
            color = bcolors.FAIL
        print(f"{f[0].ljust(25)}:{color}{res}{bcolors.ENDC}")


def dbg():
    random.seed(0)
    np.random.seed(0)
    G, V = iq3.problem(2, False)
    s = G.generate_batch(5)[0]
    a = V.one_hot(s)

    print(a.shape)
    G, V = iq3.problem(2, True)
    s = G.generate_batch(5)[0]
    a = V.one_hot(s)
    print(a.shape)


if __name__=="__main__":
    run_all()
    #dbg()
