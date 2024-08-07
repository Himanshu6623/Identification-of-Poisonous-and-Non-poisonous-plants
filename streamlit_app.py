from pymongo import MongoClient
import pickle
import tensorflow as tf
import numpy as np
import cv2
import keras
import streamlit as st

client = MongoClient("mongodb+srv://himanshuparida191003:hi%40191003@projects.dvdnu49.mongodb.net/")
db=client.Plants
collection=db.Information

class_names=['African violets-Non-poisonous',
 'Aloe vera-medicinal plant',
 'Basil-medicinal plant',
 'Caladium-poisonous',
 'basil-medicinal plant',
 'Foxglove-poisonous',
 'German chamomile-medicinal plant',
 'Lavender-medicinal plant',
 'Olender-poisonous',
 'Prayer Plant-Non-poisonous',
 'Spider Plant-Non-poisonous',
 'Sword fern-Non-poisonous']
#img=cv2.imread("/content/images.jpg", cv2.COLOR_BGR2RGB)
#cv2_imshow(img)
#cv2.waitKey(0)
st.title("Identification of Poisonous and Non-poisonous plants")
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
   uploaded_image.name = "uploaded_image.png"
   st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
   image_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
   image = cv2.imdecode(image_bytes, 1)
   model_load=pickle.load(open("/content/drive/MyDrive/Models/new_model","rb"))
   image=tf.image.resize(image,(256,256))
   input_arr = keras.utils.img_to_array(image)
   input_arr = np.array([input_arr])  # Convert single image to a batch.
   predictions = model_load.predict(input_arr)
   predicted_class = np.argmax(predictions)
   confidence = round(100 * (np.max(predictions[0])), 2)
   name = class_names[predicted_class]
   if collection.find_one({"Name":"{}".format(name)}):
       #print(name)
       document = collection.find_one({"Name":"{}".format(name)})
       name=document["Name"]
       About=document["About"]
       st.write(name,"\n")
       st.write(About)
   #st.write("This is a simple text display example.")
