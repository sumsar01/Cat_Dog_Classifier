from tensorflow.python import module
from plotting import summarize_diagnostics
from models import basic_model
from keras.preprocessing.image import ImageDataGenerator

# run test for basic model
def run_basic_model():
    
    # initilizing basic model
    model = basic_model()

    # create data generator
    data_generator = ImageDataGenerator(rescale=1.0/255.0)

    # prepare iterators
    train_ite = data_generator.flow_from_directory('data_cats_vs_dogs/train/',
        class_mode='binary',
        batch_size = 64,
        target_size=(32, 32)#(200, 200)
        )

    validation_ite = data_generator.flow_from_directory('data_cats_vs_dogs/validation/',
        class_mode='binary',
        batch_size= 64,
        target_size=(32, 32)#(200, 200)
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
    summarize_diagnostics(history, 'Basic_model')

if __name__ == '__main__':
    run_basic_model()