

def heavy_preprocessing(train_dir):
    from os import listdir
    from numpy import asarray
    from numpy import save
    from keras.preprocessing.image import load_img
    from keras.preprocessing.image import  img_to_array
    from numpy import load

    images, labels = list(), list()

    for file in listdir(train_dir):
        output = 0.0

        if file.startswith('cat'):
            output = 1.0
        
        image = load_img(train_dir + file, target_size=(200, 200))
        image = img_to_array(image)

        images.append(image)
        labels.append(output)

    images = asarray(images)
    labels = asarray(labels)

    print(images.shape, labels.shape)

    save('dogs_vs_cats_images.npy', images)
    save('dogs_vs_cats_labels.npy', labels)

    images = load('dogs_vs_cats_images.npy')
    labels = load('dogs_vs_cats_labels.npy')
    print(images.shape, labels.shape)


def light_preprocessing(train_dir):
    from os import makedirs
    from numpy.random import random
    from numpy.random import seed
    from shutil import copyfile
    from os import listdir

    dataset_dir = 'data_cats_vs_dogs/'
    subdirs = ['train/', 'validation/']

    for subdir in subdirs:
        dirlabels = ['dogs/', 'cats/']
        for dirlabel in dirlabels:
            newdir = dataset_dir + subdir + dirlabel
            makedirs(newdir, exist_ok = True)

    #setting random seed for reproducibility
    seed(0)

    #setting percentage of pictures used for validation
    val_ratio = 0.25

    #copy images to subdirs
    scource_dir = 'data/train/'

    for file in listdir(scource_dir):
        scource = scource_dir + '/' + file
        destination_dir = 'data_cats_vs_dogs/train/'

        if random(1) < val_ratio:
            destination_dir = 'data_cats_vs_dogs/validation/'
        if file.startswith('cat'):
            destination = destination_dir + 'cats/' + file
            copyfile(scource, destination)
        elif file.startswith('dog'):
            destination = destination_dir + 'dogs/' + file
            copyfile(scource, destination)