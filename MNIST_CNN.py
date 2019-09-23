import numpy as np
import numpy.random as nr
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
import keras.utils.np_utils as ku
import keras.models as models
import keras.layers as layers
from keras import regularizers
from keras.layers import Dropout
from keras.optimizers import rmsprop, Adam
plt.show()

(train_images,train_labels),(test_images,test_labels)=mnist.load_data()
train_images=train_images.reshape((60000,28,28,1)).astype('float')/255
print(train_images.shape)
print(train_images.dtype)

print(test_images.shape,test_labels.shape)
test_images=test_images.reshape((10000,28,28,1)).astype('float')/255

train_labels=ku.to_categorical(train_labels)
print(train_labels[5:,])
test_labels=ku.to_categorical(test_labels)

filepath='my_model_file.hdf5'
callbacks_list=[
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=1
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=filepath,
        monitor='val_loss',
        save_best_only=True
    )
]

nns=models.Sequential()

nns.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
nns.add(layers.MaxPooling2D(2,2))
nns.add(layers.Conv2D(64,(3,3),activation='relu'))
nns.add(layers.MaxPooling2D((2,2)))
nns.add(layers.Conv2D(64,(3,3),activation='relu'))

nns.add(layers.Flatten())

nns.add(layers.Dense(64,activation='relu',kernel_regularizer=regularizers.l2(0.01)))
nns.add(Dropout(0.5))
nns.add(layers.Dense(10,activation='softmax'))
nns.summary()


nr.seed(2356)
nns.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

nr.seed(88776)
history_s=nns.fit(train_images,train_labels,epochs=40,batch_size=128,validation_data=(test_images,test_labels),callbacks=callbacks_list,verbose=1)


def plot_loss(history):
    '''Function to plot the loss vs. epoch'''
    train_loss = history.history['loss']
    test_loss = history.history['val_loss']
    x = list(range(1, len(test_loss) + 1))
    plt.plot(x, test_loss, color='red', label='Test loss')
    plt.plot(x, train_loss, label='Training losss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')


plot_loss(history_s)


def plot_accuracy(history):
    train_acc = history.history['acc']
    test_acc = history.history['val_acc']
    x = list(range(1, len(test_acc) + 1))
    plt.plot(x, test_acc, color='red', label='Test accuracy')
    plt.plot(x, train_acc, label='Training accuracy')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Epoch')


plot_accuracy(history_s)

img = train_images[12,:,:,:]
print(img.shape)
plt.imshow(img.reshape((28,28)), cmap = 'gray')

layer_outputs = [layer.output for layer in nns.layers[:7]]
activation_model = models.Model(inputs = nns.input, outputs = layer_outputs)
activations = activation_model.predict(img.reshape(1,28,28,1))

for j in range(5):
    fig_shape = activations[j].shape
    s = fig_shape[3]/32
    fig = plt.figure(figsize=(10,6))
    for i in range(fig_shape[3]):
        ax = fig.add_subplot(s*4,8,(i+1))
        plt.imshow(activations[j].reshape((fig_shape[1],fig_shape[2],fig_shape[3]))[:,:,i], cmap='gray')
plt.show()