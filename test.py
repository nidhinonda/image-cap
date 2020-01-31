import pickle
import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from models import *


img_feat = "img_feat.h5"
enc_weights = "enc.weights"
dec_weights = "dec.weights"

embedding_dim = 256
units = 512
vocab_size = 8236
max_length = 49
attention_features_shape = 64

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

def evaluate(image):
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims( load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        #attention_plot[i] = tf.reshape(attention_weights, (-1, ))

        predicted_id = tf.argmax(predictions[0]).numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    #attention_plot = attention_plot[:len(result), :]
    return result

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)



print("Loading weights...")
encoder.load_weights(enc_weights)
decoder.load_weights(dec_weights)
image_features_extract_model = tf.keras.models.load_model(img_feat,compile=False)
print("Models loaded...")


def show_image(img):
    img_plt = plt.imshow(img)
    plt.show()

def predict(image_path):
    result0 = evaluate(image_path)
    result0 = result0[0][:len(result0[0])-1]
    result = " ".join(result0)
    return result

if __name__ == "__main__":
    while True:
        predict("bus.jpg")
        try:
            name = input("Enter file name:")
            
            if name=="":break
            result = predict(name)
            print("Predicted Caption:", result)
        except:
            print("File not found")