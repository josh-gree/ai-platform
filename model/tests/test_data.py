import pytest

import imageio

import tensorflow as tf 
import numpy as np

from model.data import data_fn, get_label, get_img

NUM_CLASSES=3
IMG_SZ=200
FEATURE_IMG_SZ=213
NUM_IMS=10

def test_get_label():
    fname = '/some/path/class_name/img_name.jpg'

    classes = np.array(['class_name','not_the_class_name'])

    want = np.array([1,0])
    got = get_label(fname,classes)

    with tf.Session() as sess:
        got = got.eval()

    np.testing.assert_array_equal(got,want)

def test_get_image(single_image):
    img = get_img(single_image)
    
    with tf.Session() as sess:
        img = img.eval()

    assert img.shape == tf.TensorShape((IMG_SZ,IMG_SZ,3))

def test_data_fn_output_shape(data_dir):
    dataset = data_fn(
        directory=str(data_dir),
        img_shape=(FEATURE_IMG_SZ,FEATURE_IMG_SZ), 
        shuffle=False
    )
    output_shapes = dataset().output_shapes

    img_shape = output_shapes[0]
    label_shape = output_shapes[1]

    assert img_shape[1:] == tf.TensorShape((FEATURE_IMG_SZ,FEATURE_IMG_SZ,3))
    assert label_shape[1:] == tf.TensorShape(NUM_CLASSES)

def test_data_fn_sz(data_dir):
    batch_size = 6
    
    dataset = data_fn(
        directory=str(data_dir),
        batch_size = batch_size, 
        shuffle=False
    )
    data_set_sz = tf.data.experimental.cardinality(dataset())
    with tf.Session() as sess:
        data_set_sz = data_set_sz.eval()

    assert data_set_sz== (NUM_CLASSES * NUM_IMS) / batch_size
def test_data_fn(data_dir):

    dataset = data_fn(
        directory=str(data_dir)
    )

    assert isinstance(dataset(),tf.data.Dataset)