import json
import numpy as np

def load_data():
    X = np.load('data/X.npy', allow_pickle=True)
    y = np.load('data/y.npy', allow_pickle=True)

    words_set = None
    with open('data/words_set.json', 'r') as f:
        words_set = json.load(f)

    return X, y, words_set
