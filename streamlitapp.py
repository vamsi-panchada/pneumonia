import streamlit as st
import cv2
import numpy as np
from tensorflow.keras import models

st.title('Pneumonia Detection Application')
st.text('Please Upload a Chest X-RAY Image to detect the Pneumonia.')

upload_file = st.file_uploader('Upload a Chest X-Ray here')

if upload_file is not None:
    print(upload_file)
    file_bytes = np.asarray(bytearray(upload_file.read()), dtype=np.uint8)
    im = cv2.imdecode(file_bytes, 1)
    im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_CUBIC)
    st.image(im, channels="BGR")
    if st.button('Predict'):
        im = im.astype(np.float32)/255.
        x = np.array([im])
        model = models.load_model('best_model.h5')
        classes = model.predict(x)
        if np.argmax(classes)==0:
            st.title(':green[NORMAL]\nYou are fine No need to Worry. 😊')
        else:
            st.title(':red[PNEUMONIA IS FOUND]\nGet Well Soon ✌🏻')
        # st.write('Image Uploaded succesfully')
