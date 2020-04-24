from __future__ import print_function
import os
import speech_recognition as sr
import sys
from deepsegment import DeepSegment
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
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
# The default language is 'en'
segmenter = DeepSegment('en')


filename = sys.argv[1]

r = sr.Recognizer()
'''
command2mp3 = ffmpeg -i Bolna.mp4 Bolna.mp3
command2wav = ffmpeg -i Bolna.mp3 Bolna.wav

os.system(command2mp3)
os.system(command2wav)
'''
audio = sr.AudioFile(filename)

# Functions to Clean Data

def remove_links(tweet):
    '''Takes a string and removes web links from it'''
    tweet = re.sub(r'http\S+', '', tweet) # remove http links
    tweet = re.sub(r'bit.ly/\S+', '', tweet) # rempve bitly links
    tweet = tweet.strip('[link]') # remove [links]
    return tweet


my_stopwords = nltk.corpus.stopwords.words('english')
word_rooter = nltk.stem.snowball.PorterStemmer(ignore_stopwords=False).stem
my_punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~@'

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
encoder = helper_dict['encoder']

#load the model
model_file_name = "./topic_classifier.h5"
model = load_model(model_file_name)

with audio as source:
	audio = r.record(source, duration=30)
	text = r.recognize_sphinx(audio)
	words = text.split(' ')
	i = 0
	#print("text: ",text)
	sentences = []
	while(i<len(words)):
		sentences.append(' '.join(words[i:i+15]))
		i=i+15
	clean_test_sentence_list = clean_data_list(sentences)
	tokenized_test_sentences = tokenizer.texts_to_sequences(clean_test_sentence_list)
	padded_test_sentences = pad_sequences(tokenized_test_sentences ,padding='post', maxlen=maxlen)
	predicts = model.predict(padded_test_sentences)

	predicts = np.array(predicts)
	counts = np.argmax(predicts, axis = 1)
	counts_arr = np.zeros((3,))
	for i in counts:
	    counts_arr[i] += 1

	label_dict = {'computer_science':"Computer Science", 'physics':'Physics', 'geography':'Geography'}
	# print("percentage of counts CS:PHY:GEO")
	# print(counts_arr/(sum(counts_arr)),"\n")
	result = encoder.inverse_transform([np.argmax(counts_arr)])[0]
	print(label_dict[result],end="")

