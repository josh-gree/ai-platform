import pytest
import imageio

import numpy as np

NUM_CLASSES=3
IMG_SZ=200
FEATURE_IMG_SZ=213
NUM_IMS=10

def create_random_jpeg(sz,name,dir):

    rand_img = np.random.rand(sz,sz,3).astype('uint8')
    imageio.imwrite(str(dir) + f'/{name}', rand_img)

@pytest.fixture(scope='function')
def data_dir(tmpdir):
    '''
    Create some random images in correct
    directory structure.
    '''
    for i in range(NUM_CLASSES):

        d = tmpdir.mkdir(f"class_{i}")

        for j in range(NUM_IMS):
            create_random_jpeg(IMG_SZ,f"sample_{j}.jpg",d)

    return tmpdir

@pytest.fixture(scope='function')
def single_image(tmpdir):
    '''
    Create some random images in correct
    directory structure.
    '''
    create_random_jpeg(IMG_SZ,f"sample.jpg",tmpdir)

    return str(tmpdir) + "/sample.jpg"