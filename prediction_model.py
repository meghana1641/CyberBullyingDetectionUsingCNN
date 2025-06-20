import numpy as np
import tensorflow as tf
import pickle
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from nltk.stem import WordNetLemmatizer
#from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import nltk
nltk.download('stopwords')
nltk.download('punkt')
import os

# Get the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'trained_model.h5')

#loading the saved model
loaded_model = tf.keras.models.load_model(model_path)

# Load the tokenizer during inference
with open('tokenizer.pkl', 'rb') as f:
    loaded_tokenizer = pickle.load(f)
    
def remove_punct(text):
  return text.translate(str.maketrans('', '',string.punctuation))

def lower(text):
    return text.lower()

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    new_text = []
    for el in word_tokenize(text):
        if not el in stop_words:
            new_text.append(el)
    return new_text

def smile_handle(word_list):
  new_word_list = []
  emoji_pattern = re.compile(r"([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])", flags=re.UNICODE)
  for word in word_list:
    if len(re.findall(emoji_pattern,word))!=0:
      if len(re.findall(emoji_pattern,word))!=len(word):
        new_word_list.append(re.sub(emoji_pattern,'',word))
      new_word_list.extend(re.findall(emoji_pattern,word))
    else:
      new_word_list.append(word)
  for i,el in enumerate(new_word_list):
    if type(el)==tuple:
      new_word_list[i] = el[1]
  return new_word_list

def lemmatize(words):
    new_words = []
    lem = WordNetLemmatizer()
    for w in words:
        new_words.append(lem.lemmatize(w))
    return new_words

#Testing
def test_example(example):
    # Preprocess the input example
    no_punctuation = remove_punct(example)
    text_lower = lower(no_punctuation)
    no_stopwords = remove_stopwords(text_lower)
    separate_words = smile_handle(no_stopwords)
    lemmatization = lemmatize(separate_words)
    testing_example = [lemmatization]
    
    testing_example = loaded_tokenizer.texts_to_sequences(testing_example)
    testing_example = pad_sequences(testing_example, maxlen=413, padding='post')

    # Make prediction using the model
    predict = loaded_model.predict(testing_example)
    predict = np.round(predict).astype(int)
    
    # Interpret the prediction
    interpretations = {
        0: "Not Cyberbullying",
        1: "Cyberbullying",
    }
    for i in interpretations.keys():
        if i == predict:
            return interpretations[i]


# Example usage
#example = "Girl bully’s as well. I’ve 2 sons that were bullied in Jr High. Both were bullied by girls. My older was bullied because he had 4ft long brown hair and a baby face. Younger was bullied cuz he hung around the nerd crowd and was an easy target. I know what u mean though! Peace"
#example='''Dou um empurrÃ£o em uma pessoa, isso se chama : - â€º empurrÃ£o. â€º â€º SOCIEDADE: bullying! http://tumblr.com/xun3xycfun'''
#prediction = test_example(example)
#print("Prediction:", prediction)

