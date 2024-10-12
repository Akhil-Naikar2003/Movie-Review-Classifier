import numpy as np 
import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

word_index=imdb.get_word_index()
reverse_word_index={value:key for key,value in word_index.items()}

model1=load_model("simple_rnn_imdb.h5")

def decode(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])


def preprossing_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review


# def predict_sentiment(review):
#     preprocess_input=preprossing_text(review)
#     predict=model1.predict(preprocess_input)
#     sentiment='Positive' if predict[0][0] >0.5 else 'Negative'
#     return sentiment,predict[0][0]

st.title("Review classifier:)")
st.write("Enter a movie review to classify it as positive or negative. ")


user_input=st.text_area("Movie Review")

if st.button('classify'):
    preprocess_input=preprossing_text(user_input)

    predict=model1.predict(preprocess_input)
    sentiment='Positive' if predict[0][0] >0.5 else 'Negative'
    


    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediciton Score : {predict[0][0]*100}')
else:
    st.write('Please enter a movie review')

