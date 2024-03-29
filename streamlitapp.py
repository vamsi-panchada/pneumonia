from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, Model
import streamlit as st
import getData
import cv2
import numpy as np

channel_axis = -1
LEARN_RATE = 1e-4

@st.cache_resource
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
    model.compile(optimizer = Adam(learning_rate = LEARN_RATE), loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])
    # print('hitting point 1')
    try:
        model.load_weights('best_model.hdf5')
    except:
        file_id = '1CIGQLKWyqDn733vIleuZqijhbcVfZbBo'
        destination = 'best_model.hdf5'
        getData.download_file_from_google_drive(file_id, destination)
        model.load_weights('best_model.hdf5')
    # print('hitting point 2')
    return model

pmodel = model()

st.title('Pneumonia Detection Application')
st.text('Please Upload a Chest X-RAY Image to detect the Pneumonia.')

imageArray = []
betaColumnArray = []

uploadedFiles = st.file_uploader('Upload Chest X-Rays', accept_multiple_files=True)

if uploadedFiles:
    st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

    for upload_file in uploadedFiles:
        uf = upload_file.read()
        file_bytes = np.asarray(bytearray(uf), dtype=np.uint8)
        im = cv2.imdecode(file_bytes, 1)
        col1, mid, col2 = st.columns([10, 1, 20])
        with col1:
             col1.image(im, channels="BGR")
             col1.write(upload_file.name)
        imageArray.append(im)
        betaColumnArray.append(col2)
        st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

if len(imageArray)>0:
    if st.button('Predict'):
        for im, col2 in zip(imageArray, betaColumnArray):
            im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_CUBIC)
            im = im.astype(np.float32)/255.
            x = np.array([im])
            classes = pmodel.predict(x)
            if np.argmax(classes)==0:
                col2.title(':green[NORMAL]\nYou are fine No need to Worry. 😊')
            else:
                col2.title(':red[PNEUMONIA IS FOUND]\nGet Well Soon ✌🏻')

imageArray.clear()
betaColumnArray.clear()