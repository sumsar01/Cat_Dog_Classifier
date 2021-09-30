from PIL.Image import TRANSPOSE
from tensorflow.python import module
from plotting import summarize_diagnostics
from models import basic_model
from keras.preprocessing.image import ImageDataGenerator


def run_model(model, IMG_SIZE, model_name):
    
    IMG_SIZE = IMG_SIZE
    
    # initilizing model
    model = model

    # create data generator
    data_generator = ImageDataGenerator(featurewise_center=True)
    data_generator.mean = [123.68, 116.779, 103.939]

    # prepare iterators
    train_ite = data_generator.flow_from_directory('data_cats_vs_dogs/train/',
        class_mode='binary',
        batch_size = 64,
        target_size=(IMG_SIZE, IMG_SIZE)
        )

    validation_ite = data_generator.flow_from_directory('data_cats_vs_dogs/validation/',
        class_mode='binary',
        batch_size= 64,
        target_size=(IMG_SIZE, IMG_SIZE)
        )

    # fit model
    history = model.fit(train_ite,
        steps_per_epoch=len(train_ite),
        validation_data=validation_ite,
        validation_steps=len(validation_ite),
        epochs=10,
        verbose=1
        )

    # evaluate model
    _, acc = model.evaluate(validation_ite, steps=len(validation_ite), verbose=1)
    print('> %.3f' % (acc * 100.0))

    #learning curves
    summarize_diagnostics(history, model_name)

if __name__ == '__main__':
    IMG_SIZE = 32
    run_model(basic_model, IMG_SIZE)