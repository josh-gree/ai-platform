'''
Construction of model that will be passed to main entrypoint to create Estimator
'''
import tensorflow as tf
import tensorflow_hub as hub

from tensorflow import keras


def model_fn(features, labels, mode, params) -> tf.estimator.EstimatorSpec:
    '''
    This function constructs model and returns and EstimatorSpec.
    '''

    inputs = tf.reshape(features["input"], [-1, params.pretrained_img_sz, params.pretrained_img_sz, 3])

    feature_extractor_url = params.pretrained_url
    feature_extractor_layer = hub.KerasLayer(
        feature_extractor_url,
        input_shape=(params.pretrained_img_sz,params.pretrained_img_sz,3)
    )

    x = feature_extractor_layer(inputs)
    x = keras.layers.Dense(params.num_hidden_units,'relu')(x)

    logits = keras.layers.Dense(params.num_classes)(x)

    if mode == tf.estimator.ModeKeys.PREDICT:

        probs = tf.nn.softmax(logits)
        classes = tf.arg_max(probs,1)

        predictions = {
            'probs':probs,
            'classes': classes
        }

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions
        )

    if mode == tf.estimator.ModeKeys.EVAL:

        labels = tf.reshape(labels, [-1, params.num_classes]) # TODO: shouldnt need this reshape
        loss =  tf.losses.softmax_cross_entropy(onehot_labels=labels,logits=logits)

        int_labels = tf.arg_max(labels,1)
        probs = tf.nn.softmax(logits)
        classes = tf.arg_max(probs,1)

        accuracy = tf.metrics.accuracy(labels=int_labels,predictions=classes)
        precision = tf.metrics.precision(labels=int_labels,predictions=classes)

        metrics = {
            'accuracy' : accuracy,
            'precision' : precision
        }

        return tf.estimator.EstimatorSpec(
            tf.estimator.ModeKeys.EVAL, 
            loss=loss, 
            eval_metric_ops=metrics
        )
    
    if mode == tf.estimator.ModeKeys.TRAIN:

        labels = tf.reshape(labels, [-1, params.num_classes]) # TODO: shouldnt need this reshape
        loss =  tf.losses.softmax_cross_entropy(onehot_labels=labels,logits=logits)

        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
        )

        return tf.estimator.EstimatorSpec(
            tf.estimator.ModeKeys.TRAIN, 
            loss=loss, 
            train_op=train_op
        )


