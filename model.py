from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import to_categorical
import glob
import cv2
import numpy as np
from tensorflow.keras import layers
import os

# Mutes warnings of Tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Number of units in each digit
numUnits = 321


def loadDataset(numUnits):

    '''
    Loads Dataset

    Parameters:
        numUnits (Int) : Number of Units in each digit

    '''

    Y = np.repeat(np.arange(0,10,1), numUnits)
    X = []

    for num in range(0,10):

        images = glob.glob('digits/'+ str(num) + '/*.png')

        for img in range(0, numUnits):
        
            im = cv2.imread(images[img], 0)
            im=im.reshape((28, 28, 1))
            im = np.asarray(im)
            X.append(im)

    X = np.array(X)

    return X, Y


def splitDataset(X, Y):

    '''
    Splits Dataset in Training, Dev and Test sets

    Parameters:
        X (numpy.ndarray) : Image Set
        Y (numpy.ndarray) : Label Set

    Returns:
        X_train (numpy.ndarray) : Image Set for Training
        X_dev (numpy.ndarray) : Image Set for Validation
        X_test (numpy.ndarray) : Image Set for Testing
        Y_train (numpy.ndarray) : Label Set for Training
        Y_dev (numpy.ndarray) : Label Set for Validation
        Y_test (numpy.ndarray) : Label Set for Testing

    '''

    from sklearn.model_selection import train_test_split
    X_train, X_dev, Y_train, Y_dev = train_test_split(X, Y, test_size = 0.3)

    from sklearn.model_selection import train_test_split
    X_dev, X_test, Y_dev, Y_test = train_test_split(X_dev, Y_dev, test_size = 0.5)

    return X_train, X_dev, X_test, Y_train, Y_dev, Y_test


def convBlock(X_prev, nf=64, f=3, F=2, s=2, r=0.3):

    '''
    Returns a set of layers

    Parameters:
        X_prev (tf.Operation) : Previous Node in the TensorFlow Graph Model
        nf (int) : Number of Filters in Conv2D Layer
        f (int) : Kernel Size for Conv2D Layer
        F (int) : Kernel Size for 2D Max Pooling Layer
        s (int) : Stride for 2D Max Pooling Layer
        r (float) : Dropout Rate

    Returns:
        X (tf.Operation) : New Node in the Tensorflow Graph Model
    
    '''


    X=layers.Conv2D(nf, (f, f), strides=1, padding='same', kernel_initializer = glorot_uniform(seed = 0), kernel_regularizer='l2')(X_prev)
    X=layers.BatchNormalization(axis=3)(X)
    X=layers.Activation('relu')(X)
    X=layers.Dropout(rate=r)(X)
    X=layers.Conv2D(nf, (f, f), strides=1, padding='same', kernel_initializer = glorot_uniform(seed = 0), kernel_regularizer='l2')(X)
    X=layers.BatchNormalization(axis=3)(X)
    X=layers.Activation('relu')(X)
    X=layers.MaxPooling2D(((F, F)), strides=s)(X)
    X=layers.Dropout(rate=r)(X)

    return X



def networkModel(size):

    '''
    Creates a Convolutional Model
    
    Parameters:
        size (Tuple) : Size of the Input Data
    
    Returns:
        model (tf.kearas.Model) : ML model

    '''


    Input=layers.Input(size)
    X=convBlock(Input)
    X=convBlock(X)
    X=layers.Flatten()(X)
    Output=layers.Dense(10, activation='softmax')(X)
    model=Model(inputs=Input, outputs=Output)
    return model


def saveModel(model, path):

    '''
    Saves model to the specified path

    Parametes:
        model (tk.keras.Model) : ML model
        path (String) : Path
    
    '''

    model.save(path)


X, Y=loadDataset(numUnits)

X_train, X_dev, X_test, Y_train, Y_dev, Y_test=splitDataset(X,Y)

# Convert Labels in Y to their categorical forms
Y_train = to_categorical(Y_train)
Y_dev = to_categorical(Y_dev)
Y_test = to_categorical(Y_test)

# Delete X and Y
del X, Y

model = networkModel(X_train.shape[1:])
batch_size = X_train.shape[0]//4
epochs = 150
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs)
model.evaluate(X_dev, Y_dev)

# Confusion Matrix
Y_conf = np.argmax(Y_test, axis=-1)
Y_pred = model.predict(X_test)
Y_pred_conf = np.argmax(Y_pred, axis=-1)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_conf, Y_pred_conf)

save_path="./digitModel"
saveModel(model, save_path)