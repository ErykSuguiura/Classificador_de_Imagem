#!/usr/bin/python3

#======================================================================================#
#                                      IMPORTS
#======================================================================================#
import flask
from flask import Flask,render_template,url_for,request
import pickle
import base64
import numpy as np
import cv2
import tensorflow as tf

from PIL import Image, ImageChops

#======================================================================================#
#                                      CODIGO
#======================================================================================#

#Initialize the useless part of the base64 encoded image.
init_Base64 = 21;

#Our dictionary
label_dict = {0:'Banana', 1:'Abacaxi', 2:'Pizza', 3:'Hamburger', 4:'Uva', 5:'Bolo'}


tf.compat.v1.disable_eager_execution()
#Initializing the Default Graph (prevent errors)

sess = tf.compat.v1.Session()

graph = tf.compat.v1.get_default_graph()

# Use pickle to load in the pre-trained model.
#with open(f'model_cnn.pkl', 'rb') as f:
#        model = pickle.load(f)

#Initializing new Flask instance. Find the html template in "templates".
app = flask.Flask(__name__, template_folder='templates')

#First route : Render the initial drawing template
@app.route('/')
def home():
	return render_template('draw.html')


#Second route : Use our model to make prediction - render the results page.
@app.route('/predict', methods=['POST'])
def predict():

        global sess
        tf.compat.v1.keras.backend.set_session(sess)

        global graph
        with graph.as_default():

            with open(f'model_cnn.pkl', 'rb') as f:
                model = pickle.load(f)
                
            if request.method == 'POST':

                    final_pred = None
                    #Preprocess the image : set the image to 28x28 shape
                    #Access the image
                    draw = request.form['url']
                    #Removing the useless part of the url.
                    draw = draw[init_Base64:]
                    #Decoding
                    draw_decoded = base64.b64decode(draw)
                    image = np.asarray(bytearray(draw_decoded), dtype="uint8")

                    img_save = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
                    cv2.imwrite('./static/result.jpg', img_save)

                    inv_img = ImageChops.invert(Image.open('./static/result.jpg'))
                    inv_img.save('./static/inv_result.jpg')
                    
                    image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

                    #Resizing and reshaping to keep the ratio.
                    resized = cv2.resize(image, (28,28), interpolation = cv2.INTER_AREA)
                    vect = np.asarray(resized, dtype="uint8")
                    vect = vect.reshape(1, 28, 28, 1).astype('float32')
                    
                    #Launch prediction
                    my_prediction = model.predict(vect)
                    #Getting the index of the maximum prediction
                    index = np.argmax(my_prediction[0])
                    #Associating the index and its value within the dictionnary
                    final_pred = label_dict[index]

        return render_template('results.html', prediction =final_pred)


if __name__ == '__main__':
	app.run(debug=True)
