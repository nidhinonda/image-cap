from flask import * 
from test import *
import werkzeug
import base64

app = Flask(__name__) 


@app.route('/request',methods=["POST"])
def req():
    print("Requesting...")
    # get image
    image_file = request.files['image']
    filename = werkzeug.utils.secure_filename(image_file.filename)
    image_file.save(filename)

    # caption it
    res = predict(filename)
    return res
    

@app.route('/')
def home():
    print("Req hello")
    return "Hello world"

if __name__ == "__main__":
    app.run(host= '0.0.0.0',debug = True,port=3999)