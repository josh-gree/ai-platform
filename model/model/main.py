'''
Main entrypoint for training, evaluation and prediction.
'''
from time import time
import click

import tensorflow as tf

from data import data_fn, get_num_classes
from model import model_fn,  Params
from utils import SERVING_FUNCTIONS

@click.command()
@click.option('--train_dir', default='./data/train', help='Location of training data.')
@click.option('--eval_dir', default='./data/eval', help='Location of evaluation data.')
@click.option('--batch_size', default=1, help='Number of samples per batch.')
@click.option('--epochs', default=1, help='Number of epochs to train for.')
@click.option('--num_hidden_units', default=100, help='Number of units in hidden layer')
@click.option('--learning_rate', default=0.001, help='Learning rate to use with Adam optimiser.')
@click.option('--pretrained_img_size', default=224, help='Image size expected by pretrained features')
@click.option('--pretrained_url', default='https://tfhub.dev/google/imagenet/resnet_v1_50/feature_vector/3', help='URL for pretrained features')
@click.option('--pretrained_tag', default=None, help='Tag for pretrained features')
@click.option('--model_name', default='model', help='Name for model. Export will be to directory {name}_{timestamp}.')
@click.option('--warm_start_dir', help='Directory from which to warm start')
def main(
    train_dir,
    eval_dir,
    batch_size,
    epochs,
    learning_rate,
    num_hidden_units,
    pretrained_img_size,
    pretrained_url,
    pretrained_tag,
    model_name,
    warm_start_dir
    ):
    # create datasets
    train_data = data_fn(
        directory=train_dir,
        batch_size=batch_size,
        img_shape=(pretrained_img_size,pretrained_img_size),
        shuffle=True
    )
    eval_data = data_fn(
        directory=eval_dir,
        batch_size=batch_size,
        img_shape=(pretrained_img_size,pretrained_img_size)
    )

    # create estimator
    num_classes = get_num_classes(train_dir)
    params = Params(
        pretrained_img_sz=pretrained_img_size,
        pretrained_url=pretrained_url,
        num_hidden_units=num_hidden_units,
        num_classes=num_classes,
        batch_size=batch_size,
        learning_rate=learning_rate,
        pretrained_tag=pretrained_tag
    )
    model_name = f"{model_name}_{int(time())}"
    estimator = tf.estimator.Estimator(
        model_fn=model_fn, 
        params=params,
        model_dir=model_name,
        warm_start_from=warm_start_dir
    )

    # Steps per epoch
    data_sz_train = tf.data.experimental.cardinality(train_data())
    data_sz_eval = tf.data.experimental.cardinality(eval_data())

    with tf.Session() as sess:
        data_sz_train = data_sz_train.eval()
        data_sz_eval = data_sz_eval.eval()
    
    initial_metrics = estimator.evaluate(eval_data, steps=data_sz_eval)
    print("Inital untrained metrics")
    print(initial_metrics)
    
    # Training
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        estimator.train(train_data, steps=data_sz_train)
        metrics = estimator.evaluate(eval_data, steps=data_sz_eval)
        print(metrics)

    # Export model for deployment
    estimator.experimental_export_all_saved_models(
        export_dir_base=f'{model_name}/exported',
        input_receiver_fn_map={
            tf.estimator.ModeKeys.PREDICT : SERVING_FUNCTIONS['JSON'](params)
        }
    )

if __name__ == '__main__':

    tf.logging.set_verbosity('ERROR') 

    main()