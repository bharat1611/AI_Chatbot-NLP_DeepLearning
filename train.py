import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Datasset, Dataloader


with open('intents.json', 'r') as f:
    intents = json.load(f);

# Tokenize
all_words = []
tags = []
xy = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

# Lowercase + Stemming + Ignore Punctuations:
ignore_words = ['?', '!', '.',',']
# print(all_words)
all_words = [stem(w) for w in all_words if w not in ignore_words]
# print(all_words)
# sorting and removing duplicates :  
all_words = sorted(set(all_words))
tags = sorted(set(tags))
# print(all_words)
# print(tags)

# Creating Bag of words and Training Data
X_train = []
Y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tags)
    # Cross Entropy Loss
    Y_train.append(label) 

X_train = np.array(X_train)
Y_train = np.array(Y_train)

