# import resource
import glob
import gc








from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing import sequence
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.recurrent import LSTM
from keras.layers.pooling import GlobalAveragePooling1D
from keras import Model
from keras.layers import average
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import plot_model, np_utils
import h5py
import sys
import numpy as np
import matplotlib
import random
matplotlib.use('AGG')

import matplotlib.pyplot as plt
import random
import os
import argparse
from PIL import Image


batch_size = 8
num_classes = 3
epochs = 30
frames = 5 # The number of frames for each sequence
input_shape = [100 , 100, 3]

def build_rgb_model2():
    model = Sequential()

    model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same'), input_shape=(frames, 120, 180, 3)))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Conv2D(32, (3, 3))))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Conv2D(32, (3, 3))))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Conv2D(32, (3, 3))))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Dropout(0.25)))
    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(512)))
    model.add(TimeDistributed(Dense(32, name="first_dense_rgb")))
    model.add(LSTM(20, return_sequences=True, name="lstm_layer_rgb"));
    model.add(TimeDistributed(Dense(num_classes), name="time_distr_dense_one_rgb"))
    model.add(GlobalAveragePooling1D(name="global_avg_rgb"))
    return model
def build_rgb_model():
    model = Sequential()
    model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu' ), input_shape=(frames, 120, 180, 4)))


#model.add(TimeDistributed(Activation('relu')))

   # model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(frames, 120, 180, 3))))
    model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')))
    model.add(TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')))
    model.add(TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Dropout(0.25)))
    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(1024)))

    model.add(TimeDistributed(Dense(32, name="first_dense_rgb")))

    model.add(LSTM(20, return_sequences=True, name="lstm_layer_rgb"));

    model.add(TimeDistributed(Dense(num_classes), name="time_distr_dense_one_rgb"))
    model.add(GlobalAveragePooling1D(name="global_avg_rgb"))

    return model


def build_flow_model():
    model = Sequential()
    model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same',activation='relu'), input_shape=(frames, 120, 180, 2)))
   # model.add(TimeDistributed(Activation('relu')))

    #model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(frames, 120, 180, 2))))
    model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')))
    model.add(TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')))
    model.add(TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Dropout(0.25)))
    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(1024)))

    model.add(TimeDistributed(Dense(32, name="first_dense_flow")))

    model.add(LSTM(20, return_sequences=True, name="lstm_layer_flow"));

    model.add(TimeDistributed(Dense(num_classes), name="time_distr_dense_one_flow"))
    model.add(GlobalAveragePooling1D(name="global_avg_flow"))

    return model


def build_model():
    rgb_model = build_rgb_model()
    flow_model = build_flow_model()



    out=average([rgb_model.output, flow_model.output])
    model=Model([rgb_model.input,flow_model.input], out)

    #model.add(add([rgb_model, flow_model]))


    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    plot_model(model, to_file='model/cnn_lstm.png')

    return model


def batch_iter(split_file):
    split_data = np.genfromtxt(split_file, dtype=None, delimiter=",")
    total_seq_num = len(split_data)

    ADRi = "720IPPP/UCF3"
    split_data2 = np.genfromtxt("C.txt", dtype=None, delimiter=",")
    # num_batches_per_epoch = int(((int(split_data2[4]) - 1) / frames - 1) / batch_size) - 500
    num_batches_per_epoch = int(((int(split_data2[4]) - 1) / frames - 1) / batch_size)
    #Error ocour in this line for bad argument
    # if split_file=='C_train.txt':
    #     num_batches_per_epoch = int(((int(split_data2[4])-1)/frames - 1) / batch_size)-500
    # else:
    #     num_batches_per_epoch = int(((int(split_data2[5])-1)/frames - 1) / batch_size)-400

    indices2=[]

    def data_generator():
        p = 0
        while 1:
            indices = np.random.permutation(np.arange(total_seq_num))
            t=0
            for j in range(total_seq_num):
                for k in range(int(split_data[j][1]/frames)):
                    indices2.append (j*1000+k*frames)
                    t=t+1
            indices3 = np.random.permutation(np.arange(indices2.__len__()))
            for batch_num in range(num_batches_per_epoch): # for each batch
                start_index = batch_num * batch_size
                # end_index = ((batch_num + 1) * batch_size) -1#####THe error ocour in this line
                end_index = ((batch_num + 1) * batch_size)#####THe error ocour in this line

                RGB = []
                FLOW = []
                Y = []
                for i in range(start_index, end_index): # for each sequence
                    print("I is {}".format(i))
                    # ii=int(indices3[i]/1000) # seqnumber
                    ii=random.randint(0,len(indices)-1)
                    image_dir = split_data[indices[ii]][0].decode("UTF-8")
                    seq_len = int(split_data[indices[ii]][1])
                    y = int(split_data[indices[ii]][2])

                    # To reduce the computational time, data augmentation is performed for each frame
                    # jj= min( int(indices3[i]/1000), seq_len-frames-1)
                    jj = min(ii, seq_len - frames - 1)
                    augs_rgb = []
                    augs_flow = []
                    for j in range(jj,jj+frames): # for each frame
                        frame = j
                        hf = h5py.File(image_dir+".h5", 'r')
                        im = hf.get(str(frame))
                        rgb =  im[:, :, :, (3,4,5,6)].transpose(0,3,1,2)
                        t=np.concatenate([rgb],axis=0)
                        augs_rgb.append(t)

                        # flow image
                        flow_x=im[:, :, :, 1]
                        flow_y = im[:, :, :, 2]

                        flow_x_flip = - np.flip(flow_x,2) # augmentation
                        flow_y_flip =   np.flip(flow_y,2) # augmentation

                        flow = np.concatenate([flow_x, flow_y], axis=0)
                        flow_flip = np.concatenate([flow_x_flip, flow_y_flip], axis=0)
                        #tt=np.concatenate([flow[None,:,:,:], flow_flip[None,:,:,:]], axis=0)
                        tt = np.concatenate([flow[None, :, :, :]], axis=0)
                        augs_flow.append(tt)

                    augs_rgb = np.array(augs_rgb).transpose((1, 0, 3, 4, 2))
                    augs_flow = np.array(augs_flow).transpose((1, 0, 3, 4, 2))
                    RGB.extend(augs_rgb)
                    FLOW.extend(augs_flow)
                    Y.extend([y])

                RGB1 = np.array(RGB)
                FLOW1 = np.array(FLOW)
                Y1 = np_utils.to_categorical(Y, num_classes)
                p=p+1
                if p==61:
                    temp=0

                yield ([RGB1, FLOW1], Y1)

    return num_batches_per_epoch, data_generator()


def plot_history(history):
    # Plot the history of accuracy
    plt.plot(history.history['acc'],"o-",label="accuracy")
    #plt.plot(history.history['val_acc'],"o-",label="val_acc")
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc="lower right")
    plt.savefig("model/model_accuracy.png")

    # Plot the history of loss
    plt.plot(history.history['loss'],"o-",label="loss",)
    #plt.plot(history.history['val_loss'],"o-",label="val_loss")
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='lower right')
    plt.savefig("model/model_loss.png")


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="action recognition by cnn and lstm.")
    parser.add_argument("--split_dir", type=str, default='split')
    parser.add_argument("--dataset", type=str, default='ucf101')
    parser.add_argument("--rgb", type=int, default=1)
    parser.add_argument("--flow", type=int, default=1)
    parser.add_argument("--split", type=int, default=1)
    args = parser.parse_args()

    split_dir = args.split_dir
    dataset = args.dataset
    rgb = args.rgb
    flow = args.flow
    split = args.split

    # Make split file path
    train_split_file = "C_train.txt" #% (split_dir, dataset, split)
    test_split_file = "C_eval.txt" #% (split_dir, dataset, split)

    # Make directory
    if not os.path.exists("model"):
        os.makedirs("model")

    # Build model
    model = build_model()
    model.summary()
    print("Built model")

    # Make batches
    train_steps, train_batches = batch_iter(train_split_file)
    valid_steps, valid_batches = batch_iter(test_split_file)

    # Train model
    history = model.fit_generator(train_batches, steps_per_epoch=train_steps,
                epochs=10, verbose=1, validation_data=valid_batches,
                validation_steps=valid_steps)
    plot_history(history)
    print("Trained model")

    # Save model and weights
    json_string = model.to_json()
    open('model/cnn_lstm.json', 'w').write(json_string)
    model.save_weights('model/cnn_lstm.hdf5')
    print("Saved model")

    # Evaluate model
    score = model.evaluate_generator(valid_batches, valid_steps)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Clear session
    from keras.backend import tensorflow_backend as backend
    backend.clear_session()
