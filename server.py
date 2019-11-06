from flask import * 
from test import *
import werkzeug
from skimage import io,transform
import base64
import numpy as np

app = Flask(__name__) 

@app.route('/request',methods=["POST"])
def req():
    print("Requesting...")
    # get image
    if 'image' in request.files:
        image_file = request.files['image']
    if 'query' in request.files:
        query = request.files['query']
    filename = werkzeug.utils.secure_filename(image_file.filename)
    image_file.save(filename)

    #rotate image
    img = io.imread(filename)
    img = transform.rotate(img,angle=90)
    img = np.flipud(img)
    img = np.fliplr(img)
    io.imsave(filename,img)
    

    # caption it
    res = predict(filename)
    return res
    

@app.route('/')
def home():
    print("Req hello")
    return "Hello world"

if __name__ == "__main__":
    app.run(host= '0.0.0.0',debug = True,port=3999)