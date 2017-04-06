# example5-keras.py
import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.set_random_seed(42)
from tensorflow.python.lib.io import file_io
from datetime import datetime
import time
# import cPickle as pickle
import pickle
import argparse

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

# reset everything to rerun in jupyter
tf.reset_default_graph()

batch_size = 100
num_classes = 2
epochs = 10

N_X = 423 # len(train_x[0])
layer1_size = 32

def train_model(train_file='sentiment_set.pickle', job_dir='./tmp/example-5', **args):
    logs_path = job_dir + '/logs/' + datetime.now().isoformat()
    print('-----------------------')
    print('Using train_file located at {}'.format(train_file))
    print('Using logs_path located at {}'.format(logs_path))
    print('-----------------------')
    file_stream = file_io.FileIO(train_file, mode='r')
    x_train, y_train, x_test, y_test  = pickle.load(file_stream)
    
    x_train = x_train.toarray()
    x_test = x_test.toarray()
    
    x_train /= np.max(x_train)
    x_test /= np.max(x_test)

    print(x_train.shape, y_train.shape, 'train samples,', type(x_train[0][0]), ' ', type(y_train[0][0]))
    print(x_test.shape,  y_test.shape,  'test samples,',  type(x_test[0][0]),  ' ', type(y_train[0][0]))

    # convert class vectors to binary class matrices. Our input already made this. No need to do it again
    # y_train = keras.utils.to_categorical(y_train, num_classes) 
    # y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Dense(layer1_size, activation='relu', input_shape=(N_X,)))
    model.add(Dropout(0.2))
    # Already overfitting, no need to add this extra layer
    # model.add(Dense(layer1_size, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test)
                        )
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    model.save('model.h5')
    
    # Save model.h5 on to google storage
    with file_io.FileIO('model.h5', mode='r') as input_f:
        with file_io.FileIO(job_dir + '/model.h5', mode='w+') as output_f:
            output_f.write(input_f.read())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
      '--train-file',
      help='GCS or local paths to training data',
      required=True
    )
    parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__
    
    train_model(**arguments)
