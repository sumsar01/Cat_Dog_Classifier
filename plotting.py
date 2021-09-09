from matplotlib import pyplot
from matplotlib.image import imread

def plot_dogs(train_dir):
    # plot first few images
    for i in range(9):
        # define subplot
        pyplot.subplot(330 + 1 + i)
        # define filename
        filename = train_dir + 'dog.' + str(i) + '.jpg'
        # load image pixels
        image = imread(filename)
        # plot raw pixel data
        pyplot.imshow(image)
    # show the figure
    pyplot.show()

def plot_cats(train_dir):
    # plot first few images
    for i in range(9):
        # define subplot
        pyplot.subplot(330 + 1 + i)
        # define filename
        filename = train_dir + 'cat.' + str(i) + '.jpg'
        # load image pixels
        image = imread(filename)
        # plot raw pixel data
        pyplot.imshow(image)
    # show the figure
    pyplot.show()