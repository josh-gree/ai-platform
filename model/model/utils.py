import os
import json
from functools import partial
import tensorflow as tf

def json_serving_input_fn(params):

    def serving_input_fn():
        """This function is used to do predictions on Google Cloud when receiving a json file."""
        input_ph = tf.placeholder(tf.string, shape=[None], name='image_binary')
        images = tf.map_fn(partial(tf.image.decode_jpeg, channels=3), input_ph, dtype=tf.uint8)
        images = tf.map_fn(partial(tf.image.convert_image_dtype, dtype=tf.float32), images, dtype=tf.float32)
        images = tf.map_fn(partial(tf.image.resize, size=(params.pretrained_img_sz,params.pretrained_img_sz)), images)

        return tf.estimator.export.ServingInputReceiver(images, {'bytes': input_ph})
    
    return serving_input_fn

SERVING_FUNCTIONS = {
    'JSON': json_serving_input_fn
}