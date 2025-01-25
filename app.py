import streamlit as st 
import tensorflow as tf
import numpy as np
from PIL import Image , ImageOps 

dic = {0:'angry', 1:"disgust", 2:"fear", 3:"happy", 4:"neutral", 5:"sad", 6:"surprise"}


model = tf.keras.models.load_model("artifacts/Model.keras")


img = st.file_uploader("Upload your photo")

st.image(img)

img = Image.open(img)

button = st.button("Predict")



if len(np.array(img).shape) != 2 : 
    st.error("Invalid Image Dim , Converting to Grayscale")
    img = ImageOps.grayscale(img)

if button==True : 
    ### resize image
    img = img.resize([48,48])
    ### normalize image 
    img = np.array(img,dtype=np.float32) 
    ### prediction 
    pred = model.predict(np.expand_dims(img, axis=0))
    pred_label = dic[np.argmax(pred)]
    st.text(f"The prediction is : {pred_label}")
