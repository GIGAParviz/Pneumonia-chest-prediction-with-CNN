import os
import cv2 as cv
import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import List
from matplotlib import pyplot as plt
import streamlit as st


model = keras.models.load_model(r"C:\Users\ASUS\OneDrive\Desktop\Pnemonia chest x ray.h5")

st.set_page_config(layout='wide')

st.title('Pneumonia Diagnosis')

uploaded_file = st.file_uploader("Choose a image file")

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv.imdecode(file_bytes, 3)
    img = cv.resize(img , (256,256))
    img = np.expand_dims(img ,axis=0)
    pre = model.predict(img)
    pre = float(pre[0])
        
    st.title('Youre Image:')
    st.image(img, channels="BGR" , width=500)
    st.text(f'you have {pre*100} chance to have Pneumomnia')