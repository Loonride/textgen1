import re
import json
import numpy as np
# import nltk

books = [
    'gatsby.txt',
    'alice.txt',
    'frankenstein.txt',
    'grimm.txt',
    'sherlock.txt',
]

txt = ""
for b in books:   
    with open(f'raw_data/{b}', 'r') as f:
        txt += f.read()

# txt = txt.replace("’", "'")
# txt = txt.replace('—', '-')
txt = txt.replace('’', "'")
# txt = txt.replace('“', '"')
# txt = txt.replace('”', '"')

# tokens = nltk.word_tokenize(txt)
# print(tokens[:10000])

chars = sorted(list(set(txt)))
print(chars)

txt = txt.lower()
words = re.findall(r"mr\.|mrs\.|[a-z0-9']+|[.,!?;:]", txt)

words_set = sorted(set(words))

words_nums = []
for w in words:
    words_nums.append(words_set.index(w))
print(words_nums[:100])

seq_length = 10
dataX = []
dataY = []
for i in range(0, len(words_nums) - seq_length, 1):
    seq_in = words_nums[i:i + seq_length]
    seq_out = words_nums[i + seq_length]
    dataX.append(seq_in)
    dataY.append(seq_out)
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)

n_words_set = len(words_set)
# reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_words_set)
# one hot encode the output variable
y = np.array(dataY)
# y = np_utils.to_categorical(dataY)

print(X)
print(y)

np.ndarray.dump(X, 'data/X.npy')
np.ndarray.dump(y, 'data/y.npy')

with open('data/words_set.json', 'w') as f:
    json.dump(words_set, f)
