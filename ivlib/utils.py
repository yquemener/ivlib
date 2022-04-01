"""
I (Yves Quemener) am the only author of this software. If you are a client of my freelance activity (IV-devs)
and this lib is used in one of the works I delivered to you, consider that you are free to
use it under the same conditions as if it were public domain or CC-0, at your convenience.

Otherwise, it is released to the public under the terms of the AGPL
%load_ext autoreload
%autoreload 2

from ivlib.utils import *
git_header()
margins_begone()

"""
import datetime
import functools
import os
import time
from copy import copy as specially_imported_copy
from types import FunctionType
import random
# import tensorflow as tf
from PIL import Image
import matplotlib.pyplot
import base64
from io import BytesIO
import numpy as np
from math import *
from html import escape

try:
    import torch
    from pygit2 import Repository, discover_repository
except ImportError:
    pass

try:
    from IPython.core.display import HTML, display
except ImportError:
    # We may not be in a jupyter notebook
    pass


def timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def datestamp():
    return datetime.datetime.now().strftime("%Y%m%d")


def pprint(l, indent=0):
    if isinstance(l, list) or isinstance(l, tuple) or isinstance(l, set):
        for ll in l:
            pprint(ll, indent+2)
    elif isinstance(l, dict):
        for k,v in l.items():
            if isinstance(v, list) or isinstance(v, tuple) or isinstance(v, set) or isinstance(v, dict):
                print(f"{indent*' '}{k}:")
                pprint(v, indent+2)
            else:
                print(f"{indent*' '}{k}: {v}")
    else:
        print(indent*' '+str(l))


def plen(l, indent=0):
    if isinstance(l, list) or isinstance(l, tuple) or isinstance(l, set):
        print(f"{' '*indent} {str(type(l))} ({str(len(l))})")
        for ll in l:
            plen(ll, indent+2)
    elif isinstance(l, dict):
        print(f"{' '*indent} {str(type(l))} ({str(len(l.keys()))})")
        for k,v in l.items():
            if isinstance(k, list) or isinstance(k, tuple) or isinstance(k, set) or isinstance(k, dict):
                print(f"{indent*' '}{k}:")
                plen((v, indent+2))


def git_header():
    repo_hash = 0
    print("============ Git Header ===============")
    try:
        repo = Repository(discover_repository(os.getcwd()))
        print("Repository:", repo.remotes[0].url)
        t = repo[repo.head.target].commit_time
        print("Last commit time:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t)))
        print("Last commit hash:", repo[repo.head.target].hex)
        repo_hash = repo[repo.head.target].hex
    except IndexError:
        print("No git repo found for this project")
    except AttributeError:
        print("No git repo found for this project")
    try:
        repo = Repository(discover_repository(os.getcwd() + "/ivlib"))
        if repo_hash != repo[repo.head.target].hex:
            print("ivlib version:")
            t = repo[repo.head.target].commit_time
            print("Last commit time:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t)))
            print("Last commit hash:", repo[repo.head.target].hex)
    except IndexError:
        print("No ivlib repo found for this project")
    except AttributeError:
        print("No ivlib repo found for this project")
    try:
        print(f"TensorFlow version: {tf.version.VERSION} "
              f"{'(GPU)' if len(tf.config.experimental.list_physical_devices('GPU')) else '(CPU)'}")
    except (ImportError, NameError):
        pass


# copying the @patch decorators from fastai
def copy_func(f):
    """Copy a non-builtin function (NB `copy.copy` does not work for this)"""
    if not isinstance(f, FunctionType):
        return specially_imported_copy(f)
    fn = FunctionType(f.__code__, f.__globals__, f.__name__, f.__defaults__, f.__closure__)
    fn.__dict__.update(f.__dict__)
    return fn


def patch_to(cls, as_prop=False):
    """Decorator: add `f` to `cls`"""
    if not isinstance(cls, (tuple, list)):
        cls = (cls,)

    def _inner(f):
        for c_ in cls:
            nf = copy_func(f)
            # `functools.update_wrapper` when passing patched function to `Pipeline`, so we do it manually
            for o in functools.WRAPPER_ASSIGNMENTS: setattr(nf, o, getattr(f, o))
            nf.__qualname__ = f"{c_.__name__}.{f.__name__}"
            setattr(c_, f.__name__, property(nf) if as_prop else nf)
        return f

    return _inner

from inspect import currentframe

def get_linenumber():
    cf = currentframe()
    return cf.f_back.f_lineno

def patch(f):
    """Decorator: add `f` to the first parameter's class (based on f's type annotations)"""
    cls = next(iter(f.__annotations__.values()))
    return patch_to(cls)(f)


def patch_property(f):
    """Decorator: add `f` as a property to the first parameter's class (based on f's type annotations)"""
    cls = next(iter(f.__annotations__.values()))
    return patch_to(cls, as_prop=True)(f)


# Class that creates balanced minibatches between good and bad inputs
class BalancedBatchMaker:
    def __init__(self, allsamples):
        self.good = set()
        self.bad = list()
        self.allsamples = allsamples

    def start_epoch(self, debug=False):
        if debug:
            print(f"Pool: good {len(self.good)} bad {len(self.bad)}")
        random.shuffle(self.bad)

    # Returns a balanced batch, trying to obey the ratio.
    # It tries to put at least one bad sample and up to the required amount.
    # If there is not enough good samples, it will complete with bad samples
    # If it can't put any bad or if there are not enough good samples to fulfill the request,
    # it returns None which is sign to start a new epoch
    def make_batch(self, size=32, ratio_good=0.5):
        batch = list()
        num_bad = int(size * ratio_good)
        num_good = size - num_bad
        if len(self.bad) == 0:
            return None
        if len(self.bad) < num_bad:
            num_bad = len(self.bad)
            num_good = size - num_bad
        if len(self.good) < num_good:
            if len(self.bad) < size - len(self.good):
                return None
            else:
                num_good = len(self.good)
                num_bad = size - num_good
        for i in range(num_bad):
            ind = random.randint(0, len(self.bad) - 1)
            bad_ind = self.bad.pop(ind)
            batch.append(self.allsamples[bad_ind] + (bad_ind,))
        good_tuple = list(self.good)
        for i in range(num_good):
            ind = random.randint(0, len(good_tuple) - 1)
            good_ind = good_tuple.pop(ind)
            self.good.remove(good_ind)
            batch.append(self.allsamples[good_ind] + (good_ind,))
        random.shuffle(batch)
        return batch


# Returns a 2x2 rot matrix
def rot_mat(angle):
    cosa = cos(angle)
    sina = sin(angle)
    mat = np.array([[cosa, sina],
                    [-sina, cosa]])
    return mat


# Returns a "smart" dir of objects, with a preview of their content
def fdir(obj):
    s = "<table>\n"
    for n in dir(obj):
        if not n.startswith("__"):
            try:
                member = getattr(obj, n)
            except Exception:
                member = None
            s += f"<tr><td>{n}</td><td style='text-align:left'>{escape(repr(type(member)))}"
            if repr(type(member)) != "<class 'method'>":
                try:
                    result = escape(repr(getattr(obj, n))[:100])
                except Exception:
                    result = "[error]"
                s += f"&nbsp;:&nbsp;{result}</td></tr>\n"
            else:
                s += "</td></tr>\n"
    s += "</table>"
    display(HTML(s))


def margins_begone():
    from IPython.core.display import display, HTML
    display(HTML("""
    <style>
         #notebook { padding-top:0px !important; }
         .container { width:100% !important; }
         .end_space { min-height:0px !important; }
         div.prompt { min-width:0px; }
         .prompt { min-width:0px; }
         .output_area { background-color:#eeeeee;}
      </style>"""))


def to_one_hot(lst):
    values = dict()
    ind = 0
    for item in lst:
        if item not in values.keys():
            ind += 1
            values[item] = ind
    result = np.zeros((len(lst), ind+1))
    for i, item in enumerate(lst):
        result[i][values[item]]=1
    return result, values


def normalize(lst):
    r = np.array(lst)
    return (r - r.min()) / (r.max() - r.min()), (r.min(), r.max())

# Needed for things like latitude/longitude that can't be stored in small floats with good precision
# Slower
def normalize_high_precision(lst):
    r = np.array(lst)
    # mean = r.mean()
    mi = r.min()
    ma = r.max()
    delta = ma-mi
    new_lst = [(x-mi)/delta for x in lst]
    return np.array(new_lst), (mi, ma)
    # return (r - mi) / (ma - mi), (mi, ma)


def denormalize(lst, mi, ma):
    r = np.array(lst)
    return r*(ma-mi)+mi


def imdisp(arr, rotate=False, zoom=(4, 4), palette=None):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().numpy()
    return display(get_img(arr, rotate, zoom, palette))


def maze_img(maze, prob=[]):
    if isinstance(maze, torch.Tensor):
        maze = maze.detach().numpy()
    maze = 1 - maze
    if len(maze.shape)==2:
        maze = np.expand_dims(maze, 2)
    if maze.shape[2]==1:
        maze = np.tile(maze, (1,1,3))
    start = dest = dir = None
    for p in prob:
        if p[0] == "start":
            start = [int(coord) for coord in p[2].split("_")[1:]]
        if p[0] == "destination":
            dest = [int(coord) for coord in p[2].split("_")[1:]]
        if p[0] == "direction":
            dir = [int(coord) for coord in p[2].split("_")[1:]]
    print(start, dest, dir)
    if start:
        maze[start[0], [start[1]], 0] = 1
        maze[start[0], [start[1]], 1] = 0
        maze[start[0], [start[1]], 2] = 0
    if dest:
        maze[dest[0], [dest[1]], 0] = 0
        maze[dest[0], [dest[1]], 1] = 0
        maze[dest[0], [dest[1]], 2] = 1
    if dir:
        maze[dir[0], [dir[1]], 0] = 0
        maze[dir[0], [dir[1]], 1] = 1
        maze[dir[0], [dir[1]], 2] = 0
    return maze


def immaze(maze, prob=[]):
    imdisp(maze_img(maze, prob), zoom=(10,10))


def immaze_html(maze, prob=[], size=(100,100)):
    img = Image.fromarray(np.uint8(maze_img(maze, prob)*255))
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('ascii')
    html = f"<img width={size[0]} height={size[1]} src='data:image/png;base64,{img_str}'>"
    return html


def get_img(arr_, rotate=False, zoom=(4, 4), palette=None):
    # arr[arr > 50] = 50
    # arr[arr < -50] = 50
    # arr = 1. / (1. + np.exp(-arr))
    arr=np.tanh(arr_)
    if len(arr.shape)==1:
        arr=np.reshape(arr, (1, -1))
    if len(arr.shape)==2:
        arr = np.reshape(arr, (arr.shape[0], arr.shape[1], 1))
        arr = np.tile(arr, (1, 1, 3))

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i][j][0] < 0:
                arr[i][j][0] = 0
                arr[i][j][1] = 0
                arr[i][j][2] = abs(arr[i][j][2])

    width = arr.shape[0]
    height = arr.shape[1]
    width *= zoom[0]
    height *= zoom[1]
    if palette is None:
        arr = np.uint8(arr * 255.0)
    else:
        col_img = list()
        for y in range(arr.shape[0]):
            row = list()
            for x in range(arr.shape[1]):
                row.append(palette[arr[y][x]])
            col_img.append(row)
        arr = np.uint8(np.array(col_img))
    if rotate:
        arr = np.swapaxes(arr, 0, 1)
        height, width = width, height
    return Image.fromarray(arr).resize((height, width), resample=Image.NEAREST)


def imrow(imglist, palette=None, zoom=(4,4)):
    fig = matplotlib.pyplot.figure(figsize=(15, 8))
    for x, img in enumerate(imglist):
        img = get_img(np.array(img), zoom=zoom, palette=palette)
        fig.add_subplot(1, len(imglist), x+1)
        matplotlib.pyplot.imshow(img)
        matplotlib.pyplot.axis('off')


class JsonUnrolled:
    def __init__(self):
        self.items = list()
        pass

    def __repr__(self):
        return repr(self.items)


def to_obj(d):
    if type(d) is dict:
        o = JsonUnrolled()
        for k, v in d.items():
            setattr(o, k, to_obj(v))
            o.items.append(k)
        return o
    elif type(d) is list:
        o = list()
        for dd in d:
            o.append(to_obj(dd))
        return o
    else:
        return d


