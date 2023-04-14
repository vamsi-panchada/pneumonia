import streamlit as st
import cv2
import numpy as np
# from tensorflow.keras import models
from tensorflow.keras.optimizers import Adam
import getData

from tensorflow.keras import layers, Model, backend
channel_axis = -1# if backend.image_data_format() == 'channels_first' else -1
def model():
    img_input = layers.Input(shape = (224, 224, 3))
    x = layers.Conv2D(32, (3,3),
                      padding = 'same', use_bias = False,
                      name = 'block1_conv1')(img_input)
    x = layers.BatchNormalization(axis = channel_axis, name = 'block1_bn1')(x)
    x = layers.Activation('relu', name = 'block1_act1')(x)
    x = layers.Conv2D(32, (3,3),
                      padding = 'same', use_bias = False,
                      name = 'block1_conv2')(x)
    x = layers.BatchNormalization(axis = channel_axis, name = 'block1_bn2')(x)
    x = layers.Activation('relu', name = 'block1_act2')(x)
    x = layers.MaxPooling2D((2, 2),
                            strides=(2, 2),
                            padding='same',
                            name='block1_pool')(x)

    # block 2
    x = layers.Conv2D(64, (3,3),
                      padding = 'same', use_bias = False,
                      name = 'block2_conv1')(x)
    x = layers.BatchNormalization(axis = channel_axis, name = 'block2_bn1')(x)
    x = layers.Activation('relu', name = 'block2_act1')(x)
    x = layers.Conv2D(64, (3,3),
                      padding = 'same', use_bias = False,
                      name = 'block2_conv2')(x)
    x = layers.BatchNormalization(axis = channel_axis, name = 'block2_bn2')(x)
    x = layers.Activation('relu', name = 'block2_act2')(x)
    x = layers.MaxPooling2D((2, 2),
                            strides=(2, 2),
                            padding='same',
                            name='block2_pool')(x)

    # block 3
    x = layers.Conv2D(128, (3,3),
                      padding = 'same', use_bias = False,
                      name = 'block3_conv1')(x)
    x = layers.BatchNormalization(axis = channel_axis, name = 'block3_bn1')(x)
    x = layers.Activation('relu', name = 'block3_act1')(x)
    x = layers.Conv2D(128, (3,3),
                      padding = 'same', use_bias = False,
                      name = 'block3_conv2')(x)
    x = layers.BatchNormalization(axis = channel_axis, name = 'block3_bn2')(x)
    x = layers.Activation('relu', name = 'block311_act2')(x)
    x = layers.MaxPooling2D((3, 3),
                            strides=(3, 3),
                            padding='same',
                            name='block3_pool')(x)

    x = layers.Conv2D(256, (3,3),
                      padding = 'same', use_bias = False,
                      name = 'block31_conv1')(x)
    x = layers.BatchNormalization(axis = channel_axis, name = 'block31_bn1')(x)
    x = layers.Activation('relu', name = 'block31_act1')(x)
    x = layers.Conv2D(128, (3,3),
                      padding = 'same', use_bias = False,
                      name = 'block31_conv2')(x)
    x = layers.BatchNormalization(axis = channel_axis, name = 'block31_bn2')(x)
    x = layers.Activation('relu', name = 'block31_act2')(x)
    x = layers.MaxPooling2D((3, 3),
                            strides=(3, 3),
                            padding='same',
                            name='block31_pool')(x)

  # block 4
    x = layers.Conv2D(1024, (3,3),
                      padding = 'same', use_bias = False,
                      name = 'block41_conv1')(x)
    x = layers.BatchNormalization(axis = channel_axis, name = 'block41_bn1')(x)
    x = layers.Activation('relu', name = 'block41_act1')(x)
    x = layers.Conv2D(512, (3,3),
                      padding = 'same', use_bias = False,
                      name = 'block41_conv2')(x)
    x = layers.Dropout(0.5, name = 'block4_dropout')(x)
    x = layers.BatchNormalization(axis = channel_axis, name = 'block4_bn2')(x)
    x = layers.Activation('relu', name = 'block4_act2')(x)
    x = layers.MaxPooling2D((3, 3),
                            strides=(3, 3),
                            padding='same',
                            name='block4_pool')(x)
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(512, activation='relu', name='fc1')(x)
    x = layers.Dense(1024, activation='relu', name='fc11')(x)
    x = layers.Dense(512, activation='relu', name='fc3')(x)
    x = layers.Dense(512, activation='relu', name='fc4')(x)
    x = layers.Dense(256, activation='relu', name='fc5')(x)
    x = layers.Dense(64, activation='relu', name='fc6')(x)
    x = layers.Dense(2, activation='softmax', name='predictions')(x)
    model = Model(inputs=img_input, outputs=x, name = 'own_build_model')
    return model

pmodel = model()
# model.summary()


LEARN_RATE = 1e-4
pmodel.compile(optimizer = Adam(learning_rate = LEARN_RATE), loss = 'categorical_crossentropy',
                           metrics = ['categorical_accuracy'])

st.title('Pneumonia Detection Application')
st.text('Please Upload a Chest X-RAY Image to detect the Pneumonia.')

containerArray = []
imageArray = []
betaColumnArray = []

# https://drive.google.com/file/d/1CIGQLKWyqDn733vIleuZqijhbcVfZbBo/view?usp=sharing
file_id = '1CIGQLKWyqDn733vIleuZqijhbcVfZbBo'
destination = 'best_model.hdf5'
getData.download_file_from_google_drive(file_id, destination)
pmodel.load_weights('best_model.hdf5')

uploadedFiles = st.file_uploader('Upload Chest X-Rays', accept_multiple_files=True)

if uploadedFiles:
    st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

    for upload_file in uploadedFiles:

        # file_bytes = np.asarray(bytearray(upload_file.read()), dtype=np.uint8)
        # im = cv2.imdecode(file_bytes, 1)
        uf = upload_file.read()
        file_bytes = np.asarray(bytearray(uf), dtype=np.uint8)
        im = cv2.imdecode(file_bytes, 1)
        container = st.container()
        col1, mid, col2 = container.columns([10, 1, 20])
        with col1:
            # col1.image(cv2.resize(im, (224, 224), interpolation=cv2.INTER_CUBIC), channels="BGR")
             col1.image(im, channels="BGR")
             col1.write(upload_file.name)
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
            # model = models.load_model('best_model.h5')
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
