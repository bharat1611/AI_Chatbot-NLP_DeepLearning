import nltk
#nltk.download('punkt')
from nltk.stem.porter import PorterStemmer 
#can use different ones for different results
stemmer = PorterStemmer()
def tokenize(sentence):
    return nltk.word_tokenize(sentence);

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    pass

# a = ["Which items do you have?", "yo"]
# print(a)
# a = tokenize(a)
# print(a)

words = ["this", "that", "three"]
stemmed_words = [stem(w) for w in words]
print(stemmed_words)
