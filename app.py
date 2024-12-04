from ast import Return
import base64
from crypt import methods
from encodings import utf_8
from typing import Any
import cv2
import os
from flask import Flask , render_template, request, redirect, url_for, render_template , Response
from cv2 import blur
from unicodedata import name
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import io
import random
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


app = Flask(__name__)

# Image processing encoding function used across models

def image_processing(image_array): # This is for encoding the image to bytes, every result image should be encoded to bytes with this function.
    
    _,image_encoded = cv2.imencode('.jpg',image_array)
    img_bytes = image_encoded.tobytes()
    return base64.b64encode(img_bytes)




@app.route('/', methods=['Get',])

def hello_world():

    
    return render_template("index.html")

@app.route('/', methods=['Post'])

#Function to upload the image from the main app route

def uploadimage():
    imagefile= request.files["file-1646503495441"]
    image_path= "./images/" +imagefile.filename
    imagefile.save(image_path) 
    
    return render_template("active.html")

# Setting up global variable for location
current_folder =os.path.dirname(os.path.abspath(__file__))
print(current_folder)
image_url = '/images/57.jpg' 

### Original Image previewer and some other processing Starts Here ###

@app.route('/original', methods=['Post', 'Get'])

def showoriginal():
    # load in color image
    image = cv2.imread("/".join([current_folder,image_url]))

    #Converting the images to bytes firstly to display it
    originalphoto = image_processing(image)

    return render_template('active.html',originalphotoshow=originalphoto.decode('utf_8'))

@app.route('/preview' , methods=['Post', 'Get'])

def index():
# load in color image
    image = cv2.imread("/".join([current_folder,image_url]))
# convert to RBG
    rbgimg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#Convert to Grey scale
    grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
#Convert to Blurred
    blurred = cv2.GaussianBlur(grayimg,(299,299),cv2.BORDER_DEFAULT)
    
#Converting the images to bytes firstly to display it
    blob = image_processing(rbgimg)
    grayencode = image_processing(grayimg)
    blurred_b64=image_processing(blurred) 


    return render_template('active.html',blur_show=blurred_b64.decode('utf_8'),blobshow=blob.decode('utf_8'),grayshow=grayencode.decode('utf_8') )

### Original Image previewer and some other processing Ends Here ###


### Canny Edge Starts Here ####

@app.route('/canny' , methods=['Post', 'Get'])

def canny():
# load in color image
    image = cv2.imread("/".join([current_folder,image_url]))
# convert to RBG
    rbgimg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#Convert to Grey scale
    grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  

#Canny Edge Detection Wide n Tight

    wide = cv2.Canny(grayimg, 30, 100)
    tight = cv2.Canny(grayimg, 200, 240)
    wideeconde = image_processing(wide)
    tightencode = image_processing(tight)

 #Canny Edge Detection with threshold optimization 

    lower = 290
    upper = 200
    edges = cv2.Canny(grayimg, lower, upper)
    edgesencode = image_processing(edges)

    return render_template('active.html',wide_decode=wideeconde.decode('utf_8'),tight_decode=tightencode.decode('utf_8'),edges_decode=edgesencode.decode('utf_8') )

### Canny Edge Ends here ####

### Face Detection Starts here ####

@app.route('/facedetection' , methods=['Post', 'Get'])

def facedetection():
# load in color image
    fd_image = cv2.imread("/".join([current_folder,image_url]))
# convert to RBG
    fd_rbgimg = cv2.cvtColor(fd_image, cv2.COLOR_BGR2RGB)
#Convert to Grey scale
    fd_grayimg = cv2.cvtColor(fd_rbgimg, cv2.COLOR_BGR2GRAY)  

# load in cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# run the detector on the grayscale image
    faces = face_cascade.detectMultiScale(fd_grayimg, 1.5, 4,44)

    img_with_detections = np.copy(fd_image)   # make a copy of the original image to plot rectangle detections ontop of

    # loop over our detections and draw their corresponding boxes on top of our original image
    for (x,y,w,h) in faces:
    # draw next detection as a red rectangle on top of the original image.  
    # Note: the fourth element (255,0,0) determines the color of the rectangle, 
    # and the final argument (here set to 5) determines the width of the drawn rectangle
         cv2.rectangle(img_with_detections,(x,y),(x+w,y+h),(255,0,0),5)  

    fd_encoded = image_processing(img_with_detections)

    return render_template('active.html',fd_show=fd_encoded.decode('utf_8'))

### Face Detection End here ####


##### The below are failure attemps to add some of the other models - Please ignore or take a look :) ####
""" 
## Color Selection and Edge Detection
# image is expected be in RGB color space# image 
# convert to RBG


# image is expected be in RGB color space# image 
def select_rgb_white_yellow(image): 
    # white color mask
    lower = np.uint8([120, 120, 120])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower, upper)
    # yellow color mask
    lower = np.uint8([190, 190,   0])
    upper = np.uint8([255, 255, 255])
    yellow_mask = cv2.inRange(image, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked = cv2.bitwise_and(image, image, mask = mask)
    return masked """

""" ### Color Selection and Edge Detection Starts Here ###

@app.route('/color' , methods=['Post', 'Get'])

def color():


# load in color image
    test_image = cv2.imread("/".join([current_folder,image_url]))
    white_yellow_images = list(map(select_rgb_white_yellow, test_image))

    
#Converting the images to bytes firstly to display it
    wyi_encoded=image_processing(white_yellow_images) 


    return render_template('active.html',wyi_show=wyi_encoded.decode('utf_8') )

### Color Selection and Edge Detection Ends Here ###


### K-Means Starts here ####

@app.route('/kmeans' , methods=['Post', 'Get'])

def kmeans():

    kmeans_img = [cv2.imread(file) for file in glob.glob("images/*")]

# Change color to RGB (from BGR)
    for i in range(len(kmeans_img)):
        kmeans_img[i] = cv2.cvtColor(kmeans_img[i], cv2.COLOR_BGR2RGB)

    def plot_images(kmeans_img):
        w=15
        h=15
        fig=plt.figure(figsize=(w, h))
        columns = 3
        rows = 3

        for i in range(1, len(kmeans_img) + 1):
            k_img = kmeans_img[i-1]
            fig.add_subplot(rows, columns, i)
            plt.imshow(k_img)
    
    return k_img

    kmeans_encoded = image_processing(k_img)

    return render_template('active.html',originalphotoshow=kmeans_encoded.decode('utf_8'))

### K-Means End here #### """


if __name__ == '__main__':
    app.run(port=3000, debug=True)