import token
from xmlrpc.client import _datetime_type
import numpy as np
import nltk
#nltk.download('punkt')
from nltk.stem.porter import PorterStemmer 
#can use different ones for different results
stemmer = PorterStemmer()
def tokenize(sentence):
    return nltk.word_tokenize(sentence);

# a = ["Which items do you have?", "yo"]
# print(a)
# a = tokenize(a)
# print(a)

def stem(word):
    return stemmer.stem(word.lower())

# words = ["this", "that", "three"]
# stemmed_words = [stem(w) for w in words]
# print(stemmed_words)


def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]


    bag = np.zeros(len(all_words), dtype = np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag

# sentence = ["hello", "how", "are", "you"]
# words = ["hi", "hello","i", "you", "bye", "thank", "cool"]
# bag = bag_of_words(sentence, words)
# print(bag)
