# import streamlit as st
# import cv2
# import numpy as np
# from tensorflow.keras import models

# st.title('Pneumonia Detection Application')
# st.text('Please Upload a Chest X-RAY Image to detect the Pneumonia.')

# upload_file = st.file_uploader('Upload a Chest X-Ray here')

# if upload_file is not None:
#     print(upload_file)
#     # file_bytes = np.asarray(bytearray(upload_file.read()), dtype=np.uint8)
#     # im = cv2.imdecode(file_bytes, 1)
#     st.image(im, channels="BGR")
#     if st.button('Predict'):
#         im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_CUBIC)
#         im = im.astype(np.float32)/255.
#         x = np.array([im])
#         model = models.load_model('best_model.h5')
#         classes = model.predict(x)
#         if np.argmax(classes)==0:
#             st.title(':green[NORMAL]')
#         else:
#             st.title(':red[PNEUMONIA IS FOUND] , Get Well Soon')
        # st.write('Image Uploaded succesfully')


import streamlit as st
import cv2
import numpy as np
from tensorflow.keras import models

st.title('Pneumonia Detection Application')
st.text('Please Upload a Chest X-RAY Image to detect the Pneumonia.')

containerArray = []
imageArray = []
betaColumnArray = []

# <<<<<<< HEAD
# if upload_file is not None:
#     print(upload_file)
#     file_bytes = np.asarray(bytearray(upload_file.read()), dtype=np.uint8)
#     im = cv2.imdecode(file_bytes, 1)
#     im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_CUBIC)
#     st.image(im, channels="BGR")
#     if st.button('Predict'):
#         im = im.astype(np.float32)/255.
#         x = np.array([im])
#         model = models.load_model('best_model.h5')
#         classes = model.predict(x)
#         if np.argmax(classes)==0:
#             st.title(':green[NORMAL]\nYou are fine No need to Worry. 😊')
#         else:
#             st.title(':red[PNEUMONIA IS FOUND]\nGet Well Soon ✌🏻')
#         # st.write('Image Uploaded succesfully')
# =======

uploadedFiles = st.file_uploader('Upload Chest X-Rays', accept_multiple_files=True)

pmodel = models.load_model('best_model.h5')

if uploadedFiles:
#     ColumnArray = list(st.columns(len(uploadedFiles)))
    st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

    for upload_file in uploadedFiles:

        # file_bytes = np.asarray(bytearray(upload_file.read()), dtype=np.uint8)
        # im = cv2.imdecode(file_bytes, 1)
        file_bytes = np.asarray(bytearray(upload_file.read()), dtype=np.uint8)
        im = cv2.imdecode(file_bytes, 1)
        container = st.container()
        col1, mid, col2 = container.columns([10, 1, 20])
        with col1:
            # col1.image(cv2.resize(im, (224, 224), interpolation=cv2.INTER_CUBIC), channels="BGR")
             col1.image(im, channels="BGR")
        containerArray.append(container)
        imageArray.append(im)
        betaColumnArray.append(col2)
        st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)


if len(imageArray)>0:
    if st.button('Predict'):
        i = 0
        for container, im, col2 in zip(containerArray, imageArray, betaColumnArray):
            # container.write(i)
            # i+=1 
            # im = cv2.resize(im, (512, 512))[:,:,0]
            # im = im.reshape(1, 512, 512, 1)
            # mask = model.predict(im).reshape(512, 512)
            # im = im.reshape(512, 512)
            # im[mask==0]=0
            # im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_CUBIC).reshape(1, 224, 224, 1)
            # im = im.astype(np.float32)/255.

            im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_CUBIC)
            im = im.astype(np.float32)/255.
            x = np.array([im])
            model = models.load_model('best_model.h5')
            # classes = model.predict(x)
            
            classes = pmodel.predict(x)
            if np.argmax(classes)==0:
                # engine.say('your x-ray looks fine')
                with col2:
                    col2.title(':green[NORMAL]\nYou are fine No need to Worry. 😊')
                # text_to_speech('pneumonia is detected, better consult a doctor')
                # os.system("say 'your x-ray looks fine'")
                
            else:
                # engine.say('pneumonia is detected, better consult a doctor')
                with col2:
                    col2.title(':red[PNEUMONIA IS FOUND]\nGet Well Soon ✌🏻')
                # text_to_speech('pneumonia is detected, better consult a doctor')
                # os.system("say 'pneumonia is detected, better consult a doctor'")
                
            # engine.runAndWait()
            # engine.stop()
# >>>>>>> 63c6a23 (multi image feature added)
