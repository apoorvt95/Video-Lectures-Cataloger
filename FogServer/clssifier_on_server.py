from __future__ import print_function
import re
import pickle

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.models import load_model
from keras import layers
from keras.utils import np_utils

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import nltk




# Functions to Clean Data

def remove_links(tweet):
    '''Takes a string and removes web links from it'''
    tweet = re.sub(r'http\S+', '', tweet) # remove http links
    tweet = re.sub(r'bit.ly/\S+', '', tweet) # rempve bitly links
    tweet = tweet.strip('[link]') # remove [links]
    return tweet


my_stopwords = nltk.corpus.stopwords.words('english')
word_rooter = nltk.stem.snowball.PorterStemmer(ignore_stopwords=False).stem
my_punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@'

def clean_text(text, bigrams=False):
    text = text.replace("\n"," ")
    text = text.strip()
    #print("step 1: remove leading and ending space and new line chars\n"+text+"\n")
    text = remove_links(text)
    #print("step 2: remove links\n"+text+"\n")
    text = text.lower() # lower case
    #print("step 3:convert to lower case\n"+text+"\n")
    text_token_list = [word for word in text.split(' ')
                            if word not in my_stopwords]
    
    text = ' '.join(text_token_list)
    #print("step 4:remove stop words\n"+text+"\n")
    
    text = re.sub('['+my_punctuation + ']+', ' ', text) # strip punctuation
    #print("step 5: removing punctuation\n"+text+"\n")
    
 
    text = re.sub('([0-9]+)', '', text) # remove numbers
    #print("step 7:remove numbers\n"+text+"\n")
    text = " ".join(text.split())  #remove all spacing
    #print("step 8:remove double spaces, tab spaces\n"+text+"\n")
    # remove stopwords

    text_token_list = [word_rooter(word) if '#' not in word else word
                        for word in text.split()] # apply word rooter
    if bigrams:
        text_token_list = text_token_list+[text_token_list[i]+'_'+text_token_list[i+1]
                                            for i in range(len(text_token_list)-1)]
    text = ' '.join(text_token_list)
    return text

def clean_data_list(text_list):
    clean_data = []
    for i in text_list:
        clean = clean_text(i)
        if len(clean) > 0:
            clean_data.append(clean)
    return clean_data

def read_file_clean_data(file_name):
    new_data = []
    with open(file_name, 'r', encoding="utf8") as fp:
        for i in fp:
            l = i.replace("\n",'')
            l = ' '.join(l.split())
            l_arr = l.split('.')
            # print(l,type(l), len(l))
            if len(l_arr) > 0:
                new_data = new_data + l_arr
    return new_data


# Function to load the tokenizer

pickle_file = "classifier_helper.pkl"
with open(pickle_file, "rb") as fp:
    helper_dict = pickle.load(fp)

tokenizer = helper_dict["tokenizer"]
maxlen = helper_dict['maxlen']

#load the model
model_file_name = "./topic_classifier.h5"
model = load_model(model_file_name)

# Now open the test file and ge the cleaned up data
test_file = "./data/test.txt"
sentences = []
with open(test_file,'r', encoding="UTF-8") as fp:
    for i in fp:
        j = i.split()
        k = 0
        while k < len(j):
            #print(len(j[k:k+15]))
            sentences.append(' '.join(j[k:k+15]))
            k = k+15
        if k-15 < len(j):
            #print(len(j[k-15:]))
            sentences.append(' '.join(j[k-15:]))

clean_test_sentence_list = clean_data_list(sentences)
tokenized_test_sentences = tokenizer.texts_to_sequences(clean_test_sentence_list)
padded_test_sentences = pad_sequences(tokenized_test_sentences ,padding='post', maxlen=maxlen)
predicts = model.predict(padded_test_sentences)

predicts = np.array(predicts)
counts = np.argmax(predicts, axis = 1)
counts_arr = np.zeros((3,))
for i in counts:
    counts_arr[i] += 1

label_dict = {0:"Computer Science", 1:'Physics', 2:'Geography'}
print("percentage of counts CS:PHY:GEO")
print(counts_arr/(sum(counts_arr)),"\n")
print("label: ", label_dict[np.argmax(counts_arr)])