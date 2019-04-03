# Select processing devices
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# DEPENDECIES
import numpy as np
import os

import keras
from keras.models import Sequential, Model
from  keras.layers import Input, Dense, Conv2D, BatchNormalization, Dropout, Flatten
from  keras.layers import  Activation, Reshape, Conv2DTranspose, UpSampling2D
from  keras.optimizers import RMSprop

import  pandas
from matplotlib import pyplot as plt


# Load Data
input1 = "Data\\topic"
data = np.load(input1)
img_w, img_h = data.shape[1:3]


def discriminator_builder(width=1024, p=0.4):
    # deine inputs
    inputs = Input((img_w, img_h, 1))

    #Convolutional Layers
    conv1 = Conv2D(width*1, 5, strides=2, padding='same', activation='relu')(inputs)
    conv1 = Dropout(p)(conv1)

    conv2 = Conv2D(width * 2, 5, strides=2, padding='same', activation='relu')(conv1)
    conv2 = Dropout(p)(conv2)

    conv3 = Conv2D(width * 4, 5, strides=2, padding='same',activation='relu')(conv2)
    conv3 = Dropout(p)(conv3)

    conv4 = Conv2D(width * 8, 5, strides=1, padding='same', activation='relu')(conv3)
    conv4 = Flatten()(Dropout(p)(conv4))

    #outputlayer
    output = Dense(1, activation='sigmoid')(conv4)

    # Model Deinition
    model = Model(inputs= inputs, outputs=output)
    # model.summary()
    return model


def generator_adversarial_example(width=1024, p=0.4):
    # deine inputs
    inputs = Input((img_w, img_h, 1))

    # Convolutional Layers
    conv1 = Conv2D(width * 1, 5, strides=2, padding='same', activation='relu')(inputs)
    conv1 = Dropout(p)(conv1)

    conv2 = Conv2D(width * 2, 5, strides=2, padding='same', activation='relu')(conv1)
    conv2 = Dropout(p)(conv2)

    conv3 = Conv2D(width * 4, 5, strides=2, padding='same', activation='relu')(conv2)
    conv3 = Dropout(p)(conv3)

    conv4 = Conv2D(width * 8, 5, strides=1, padding='same', activation='relu')(conv3)
    conv4 = Flatten()(Dropout(p)(conv4))

    # outputlayer
    output = Dense(1, activation='sigmoid')(conv4)

    # Model Deinition
    model = Model(inputs=inputs, outputs=output)
    # model.summary()
    return model

    #Create Adversarial Network
def adversarial_builder():
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0004, decay=3e-8, clipvalue=1.0),
                              metrics=['accuracy'])
    # model.summary()
    return  model


#Train
def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

def train(epochs = 2000, batch = 128):

    d_metrics = []
    a_metrics = []

    running_d_loss = 0
    running_d_acc = 0
    running_a_loss = 0
    running_a_acc= 0

    for i in range(epochs):
        print(i)
        if i%100 == 0:
            print(i)
        real_data = np.reshape(data[np.random.choice(data.shape[0], batch, replace=False)], (batch, 28, 28, 1))
        fake_data = generator.predict(np.random.uniform(-1.0, -1.0, size=[batch, 100]))
        x= np.concatenate((real_data,fake_data))
        y = np.ones([2*batch, 1])
        y[batch:,:] = 0
        make_trainable(discriminator, True)
        d_metrics.append(discriminator.train_on_batch(x,y))
        running_d_loss +=d_metrics[-1][0]
        running_d_acc += d_metrics[-1][1]

        make_trainable(discriminator, False)

        y = np.ones([batch, 1])

        a_metrics.append(adversarial_model.train_on_batch(real_data, y))
        running_a_loss += d_metrics[-1][0]
        running_a_acc += d_metrics[-1][1]
        if ((i+1)%500 ==0):
            print('Epoch#{}'.format(i+1))
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, running_d_loss/i, running_d_acc/i)
            log_mesg = "%s  [A loss: %f, acc: %f]" %  (log_mesg, running_a_loss/i, running_a_acc/i)
            print(log_mesg)
    return a_metrics, d_metrics

if __name__=='__main__':
    discriminator = discriminator_builder()
    discriminator.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0008, decay=6e-8, clipvalue=1.0), metrics=['accuracy'])
    generator = generator_adversarial_example()
    adversarial_model = adversarial_builder()
    a_metrics_complete, d_metrics = train(epochs = 1000)
