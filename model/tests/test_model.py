import pytest

import tensorflow as tf  
 
import numpy as np

from collections import namedtuple

from model.model import model_fn
from model.data import data_fn

@pytest.fixture(scope='module')
def params():
    Params = namedtuple(
        'params', 
        [
            'pretrained_img_sz',
            'pretrained_url',
            'num_hidden_units',
            'num_classes',
            'batch_size'
            ]
        )

    batch_size = 10
    num_classes = 4
    num_hidden_units = 256

    pretrained_img_sz = 224
    pretrained_url='https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2'

    return Params(
        pretrained_img_sz=pretrained_img_sz,
        pretrained_url=pretrained_url,
        num_hidden_units=num_hidden_units,
        num_classes=num_classes,
        batch_size=batch_size
    )

@pytest.fixture(scope='module')
def labels(params): 

    labels = np.zeros((params.batch_size,params.num_classes))
    for i in range(params.batch_size):
        zeros = np.zeros(params.num_classes)
        ind = np.random.randint(0,params.num_classes)
        zeros[ind] = 1
        labels[i,:] = zeros 
    labels = tf.constant(labels.astype('int')) 
    return labels

@pytest.fixture(scope='module')
def features(params):  

    features = tf.constant(
        np.random.rand(params.batch_size,params.pretrained_img_sz,params.pretrained_img_sz,3).astype('float32')
    )
    return {
        'input': features
    }

def test_predict(params, features, labels):
    '''
    Check model_fn behaves as expected for PREDICT
    '''
    mode = tf.estimator.ModeKeys.PREDICT

    spec = model_fn(features,labels,mode,params)

    assert isinstance(spec,tf.estimator.EstimatorSpec)
    assert spec.predictions is not None
    assert spec.loss is None
    assert spec.train_op is None

def test_eval(params, features, labels):
    '''
    Check model_fn behaves as expected for EVAL
    '''
    mode = tf.estimator.ModeKeys.EVAL

    spec = model_fn(features,labels,mode,params)
    
    assert isinstance(spec,tf.estimator.EstimatorSpec)
    assert spec.predictions is not None
    assert spec.loss is not None
    assert spec.train_op is None

def test_train(params, features, labels):
    '''
    Check model_fn behaves as expected for TRAIN
    '''
    mode = tf.estimator.ModeKeys.TRAIN

    spec = model_fn(features,labels,mode,params)
    
    assert isinstance(spec,tf.estimator.EstimatorSpec)
    assert spec.predictions is not None
    assert spec.loss is not None
    assert spec.train_op is not None

def test_create_estimator(params, features, labels, data_dir):
    '''
    Check we can succesfully create estimator from model_fn and train 
    and evaluate for one step.
    '''
    estimator = tf.estimator.Estimator(
        model_fn=model_fn, 
        params=params
    )

    dataset = data_fn(directory=str(data_dir))
    estimator.train(lambda : dataset, steps=1)
    estimator.evaluate(lambda : dataset, steps=1)