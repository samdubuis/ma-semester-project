import os
import numpy as np
import cv2
import tensorflow as tf

def USM_DB_loader(directory, resize):
    """
    Function to load a directory of BMP images into a dataset to be used for training

    @directory : the directory path where the files are
    """
    data_dir = directory
    size = 0

    labels = []
    filenames = []
    
    for i in range(1, 124):
        for j in range(1,5):
            for file in os.listdir(data_dir+
                                   "vein"+
                                   str(i).zfill(3)+"_"+str(j)):
                if file.endswith(".jpg"):
                    filenames = filenames+[data_dir+
                                           "vein"+
                                           str(i).zfill(3)+"_"+str(j)+"/"+file]
                    labels = labels+[i]

    # print(filenames)
    # print(labels)

    filenames = tf.constant(filenames)
    labels = tf.constant(labels)
    
    print("Shape of filenames tensor : {}" .format(filenames.shape))
    print("Size of images : {}" .format(tf.image.decode_jpeg(tf.io.read_file(filenames[0])).shape))
    
    # step 2: create a dataset returning slices of `filenames`
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

    # step 3: parse every image in the dataset using `map`
    def _parse_function(filename, label):
        image_string = tf.io.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=1)
        # image_resized = tf.image.resize_with_pad(image_decoded, resize, resize)
        image = tf.cast(image_decoded, tf.float32)
        return image, label-1 #tf.one_hot(label-1, 123)

    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(10000)

    print("Data {} loaded\n" .format(directory))
    return dataset


def create_domainB(data_dir, gan_dir, brightness_adjustement, phase="train"):

    l = np.random.choice(2952, size=100, replace=False)+1
    
    print(l)
    
    if phase=="train":
        for i in l:
            # print(i)
            path = data_dir+gan_dir+"/trainA/"
            file = [j for j in os.listdir(path) if j.startswith("finger{}".format(str(i).zfill(4)))][0]
            img = tf.io.read_file(path+file)
            img = tf.io.decode_image(img, channels=1)

            tmp = tf.image.adjust_brightness(img, brightness_adjustement)
            cv2.imwrite(data_dir+gan_dir+"/trainB/"+file, tmp.numpy())
            os.remove(path+file)
        print("Domain B for training created\n")
            
            
    elif phase == "test":
        for i in l:
            # print(i)
            path = data_dir+gan_dir+"/testA/"
            file = [j for j in os.listdir(path) if j.startswith("finger{}".format(str(i).zfill(4)))][0]
            img = tf.io.read_file(path+file)
            img = tf.io.decode_image(img, channels=1)

            tmp = tf.image.adjust_brightness(img, 0.2)
            cv2.imwrite(data_dir+gan_dir+"/testB/"+file, tmp.numpy())
            os.remove(path+file)
        print("Domain B for testing created\n")


def final_stn_loader(directory, resize=256, stn=True):
    data_dir = directory
    size = 0

    labels = []
    filenames = []
    
    for file in os.listdir(data_dir):
        if file.endswith(".jpg"):
            filenames = filenames+[data_dir+file]
            tmp = file.split("_u")[1]
            tmp = tmp.split(".")[0]
            labels = labels+[int(tmp)]

    # print(filenames)
    # print(labels)

    filenames = tf.constant(filenames)
    labels = tf.constant(labels)
    
    # print("Shape of filenames tensor : {}" .format(filenames.shape))
    # print("Size of images : {}" .format(tf.image.decode_jpeg(tf.io.read_file(filenames[0])).shape))
    
    # step 2: create a dataset returning slices of `filenames`
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

    # step 3: parse every image in the dataset using `map`
    def _parse_function(filename, label):
        image_string = tf.io.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=1)
        if stn:
            image = tf.cast(image_decoded, tf.float32)
        else:
            image_resized = tf.image.resize_with_pad(image_decoded, resize, resize)
            image = tf.cast(image_resized, tf.float32)
        return image, label #tf.one_hot(label-1, 123)

    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(10000)

    print("Data {} loaded\n" .format(directory))
    return dataset