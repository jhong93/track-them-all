import json
import pickle
import base64
import gzip
from io import BytesIO
from PIL import Image
import numpy as np


def load_json(fpath):
    with open(fpath) as fp:
        return json.load(fp)


def load_gz_json(fpath):
    with gzip.open(fpath, 'rt', encoding='ascii') as fp:
        return json.load(fp)


def store_json(fpath, obj, indent=None):
    kwargs = {}
    if indent is not None:
        kwargs['indent'] = indent
    with open(fpath, 'w') as fp:
        json.dump(obj, fp, **kwargs)


def store_gz_json(fpath, obj, indent=None):
    kwargs = {}
    if indent is not None:
        kwargs['indent'] = indent
    with gzip.open(fpath, 'wt', encoding='ascii') as fp:
        json.dump(obj, fp, **kwargs)


def load_pickle(fpath):
    with open(fpath, 'rb') as fp:
        return pickle.load(fp)


def store_pickle(fpath, obj):
    with open(fpath, 'wb') as fp:
        pickle.dump(obj, fp)


def decode_png(data):
    if isinstance(data, str):
        # Densepose encodes with b64 for whatever reason
        data = base64.decodebytes(data.encode())
    else:
        assert isinstance(data, bytes)
    fstream = BytesIO(data)
    im = Image.open(fstream)
    return np.array(im)


def encode_png(data, optimize=True):
    im = Image.fromarray(data)
    fstream = BytesIO()
    im.save(fstream, format='png', optimize=optimize)
    s = base64.encodebytes(fstream.getvalue()).decode()
    return s


def load_text(fpath):
    lines = []
    with open(fpath, 'r') as fp:
        for l in fp:
            l = l.strip()
            if l:
                lines.append(l)
    return lines


def store_text(fpath, s):
    with open(fpath, 'w') as fp:
        fp.write(s)


def decode_heatmap(data):
    data = base64.decodebytes(data.encode())
    return np.load(BytesIO(data), allow_pickle=True)['h']