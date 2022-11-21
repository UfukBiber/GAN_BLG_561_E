import tensorflow as tf 
from os import path, listdir
from random import shuffle


IMAGE_FOLDER = "/home/ufuk/Desktop/img_align_celeba"

## Loading the image from path with RGB format
## Resizing the image
## Casting the image array to float32
def load_image(img_path:str, image_size:list):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels = 3)
    img = tf.image.resize(img, size = image_size, method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    img = tf.cast(img, tf.float32)
    img = (img - 127.5) / 127.5
    return img

## Loading the image from path with Grayscale format
## Resizing the image
## Casting the image array to float32
def load_image_grayscale(img_path:str, image_size:list):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels = 1)
    img = tf.image.resize(img, size = image_size, method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    img = tf.cast(img, tf.float32)
    img = (img - 127.5) / 127.5
    return img


## Loads the dataset 
def load_dataset(image_size:list, batch_size:int, num_of_image=None, is_grayscale:bool = False):
    ## Reading the paths for images
    paths = [path.join(IMAGE_FOLDER, img_path) for img_path in listdir(IMAGE_FOLDER) if img_path.endswith("jpg")]
    ## Taking the certain number of images if given.
    if (num_of_image is not None):
        shuffle(paths)
        paths = paths[:num_of_image]
    ## Creating the dataset
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ## Mapping the dataset 
    if (not is_grayscale):
        ds = ds.map(lambda img_path:load_image(img_path, image_size), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(lambda img_path:load_image_grayscale(img_path, image_size), num_parallel_calls=tf.data.AUTOTUNE)
    ## Creating the batches
    ## Prefetch creates buffers with CPU, and GPU uses this prepared data without waiting.
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return ds 