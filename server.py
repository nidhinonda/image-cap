from flask import * 
from test import *
import werkzeug
from skimage import io,transform
import base64
import numpy as np
from text_rec.text_rec import *
import tensorflow as tf
import requests

app = Flask(__name__) 


VQA_APP_URL = "http://127.0.0.1"
VQA_APP_PORT = "2222"


@app.route('/request',methods=["POST"])
def req():
    print("Requesting...")
    # Get image
    if 'image' in request.files:
        image_file = request.files['image']
    if 'request' in request.form:   
        query = request.form['request']
    
    

    filename = werkzeug.utils.secure_filename(image_file.filename)
    image_file.save(filename)

    # Rotate and Flip image
    img = io.imread(filename)
    img = transform.rotate(img,angle=90)
    img = np.flipud(img)
    img = np.fliplr(img)
    io.imsave(filename,img)
    
    return make_query_decisions(query,img,filename)

@app.route('/')
def home():
    print("Req hello")
    return "Hello world"


def make_query_decisions(query,img,filename):
    query = query.strip()
    if check_messages_in(["caption this", "caption"],query) == True:
        # Generate Caption
        print("Captioning...") 
        res = predict(filename)
        return res
    
    elif check_messages_in(["read","read this"],query) == True:
        # generate text
        print("Text Recognition...")
        res = get_text(cv2.imread(filename))
        print(res)
        return " ".join(res.split('\n'))

    else: 
        # make a call to the VQA App
        print("Executing VQA Task")
        files = {'image_file': open(filename, 'rb')}
        res = requests.post(VQA_APP_URL + ":" + VQA_APP_PORT + "/predict", files = files, data = {"question": query} )
        print("Response:",res.status_code, res, res.text)
        if res.text != "":
            return res.text
        else:
            return "Unable to process"

def check_messages_in(a,b):
    '''
    Checks if list of items in a is in single string b
    '''
    for i in a:
        if i in b:
            return True 
    return False

if __name__ == "__main__":
    app.run(host= '0.0.0.0',debug = True,port=3999)


