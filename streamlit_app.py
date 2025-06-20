# importing libraries

from ctypes import alignment
from urllib import response
import streamlit as st
import altair as alt
from PIL import Image
import numpy as np
from prediction_model import test_example

# Page title

image = Image.open('Frontpage.png')

st.image(image, use_column_width= True)

st.write('''
# Cyberbulling Tweet Recognition App

This app predicts whether the tweet is
* Cyberbullying
* Not Cyberbullying

***
''')

# Text Box
st.header('Enter Tweet ')
tweet_input = st.text_area("Tweet Input", height= 150)
print(tweet_input)
st.write('''
***
''')

# print input on webpage
st.header("Entered Tweet text ")
if tweet_input:
    tweet_input
else:
    st.write('''
    ***No Tweet Text Entered!***
    ''')
st.write('''
***
''')

# Output on the page
st.header("Prediction")
if tweet_input:
    prediction = test_example(tweet_input)
    if prediction == "Cyberbullying":
        st.image("Cyberbullying.png",use_column_width= True)
    elif prediction == "Not Cyberbullying":
        st.image("not_cyberbullying.png",use_column_width= True)
    
else:
    st.write('''
    ***No Tweet Text Entered!***
    ''')

st.write('''***''')
