import tensorflow as tf
import numpy as np

from glob import glob

def get_label(fname, classes):
    one_hot = (fname.split('/')[-2] == classes).astype(int)
    return tf.convert_to_tensor(one_hot)


def get_img(fname,img_shape=None):

    img = tf.io.read_file(fname)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    if img_shape:
        img = tf.image.resize(img,(img_shape[0],img_shape[1]))
        
    return img

def data_fn(*, directory, batch_size=1, img_shape=None, shuffle=True):

    class_dirs = glob(directory + '/*')
    classes = np.array([class_dir.split('/')[-1] for class_dir in class_dirs])
    num_class = len(class_dirs)

    files = glob(directory + '/*/*')
    
    labels = [get_label(f,classes) for f in files]
    imgs = [get_img(f,img_shape=img_shape) for f in files]

    data_set = tf.data.Dataset.from_tensor_slices((imgs,labels))

    if shuffle:
        return data_set.shuffle(len(imgs)).batch(batch_size)
    else:
        return data_set.batch(batch_size)