import streamlit as st
from tf.keras.models import load_model
import numpy as np
from PIL import Image


model = joblib.load('iris.h5')

def speciespred(seplen, sepwid, petlen, petwid):

    #reshaping feature input to be fed into the model
    arg = [[seplen, sepwid, petlen, petwid]]
    #Predicts and Returns Species
    species = model.predict(arg)[0]

    return species.upper()

st.title('Iris Flower Recognition')
st.subheader('Using Machine Learning - Decision Tree Classifier')

slen = st.slider('Sepal Length',0.0,10.0,step=0.1)
swid = st.slider('Sepal Width',0.0,10.0,step=0.1)
plen = st.slider('Petal Length',0.0,10.0,step=0.1)
pwid = st.slider('Petal Width',0.0,10.0,step=0.1)

if st.button('Predict Flower Species'):

    txt = speciespred(slen, swid, plen, pwid)
    if txt == 'Iris-Setosa'.upper():
        img = Image.open('Iris Setosa.jpeg')
    elif txt == 'Iris-Versicolor'.upper():
        img = Image.open('Iris Versicolor.jpg')
    elif txt == 'Iris-Virginica'.upper():
        img = Image.open('Iris Virginica.jpg')
    
    st.subheader("_Model Prediction_")
    st.write("Prediction:  ")
    st.title(txt)
    st.image(img)
    st.balloons()

    
    st.write("")
    st.write("")
    st.sidebar.write("**_Developed by Neha Vishwakarma_**")
    st.write("")
    st.write("**Hosted at GitHub:  @knowminni**")
    st.write("_Under Apache License 2.0_")
