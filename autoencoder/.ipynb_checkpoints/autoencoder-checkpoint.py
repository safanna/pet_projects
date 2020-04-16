import numpy as np
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import Callback

class Autoencoder:
    def __init__(self, input_dim, hid_dim, epoch=500, batch_size=10):
        self.epoch = epoch
        self.batch_size = batch_size
        self.hid_dim = hid_dim
        x = Input((input_dim,))
        dense = Dense(self.hid_dim)(x)
        encoded = Activation('relu')(dense)
        decoded = Dense(input_dim, name='decoded')(encoded)
        self.model = Model(inputs=x, outputs=[decoded, encoded])
        
    def train(self, data, learning_rate=0.001):
        train_op = RMSprop(learning_rate)
        self.model.compile(loss={"decoded": self.root_mean_squared_error}, optimizer=train_op)

        self.model.fit(data, data, epochs = self.epoch, batch_size = self.batch_size)
        self.model.save("./autoencoder.h5")
            
    @staticmethod
    def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 
        
    def test(self, data):
        pred = self.model.predict(data)
        print('input', data)
        print('compressed', pred[1])
        print('reconstructed', pred[0])
        return pred

    def classify(self, data, labels, ind=7):
        reconstructed, hidden = self.model.predict(data)
        print('data', np.shape(data))
        print('reconstructed', np.shape(reconstructed))
        loss = np.sqrt(np.mean(np.square(data - reconstructed), axis=1))
        print('loss', np.shape(loss))
        subj_indices = np.where(labels == ind)[0]
        not_subj_indices = np.where(labels != ind)[0]
        subj_loss = np.mean(loss[subj_indices])
        not_subj_loss = np.mean(loss[not_subj_indices])
        print('subj', subj_loss)
        print('not subj', not_subj_loss)
        return hidden

    
    def decode(self, encoding):
        inputs = Input((self.hid_dim,))
        outputs = self.model.get_layer('decoded')(inputs)
        model_dec = Model(inputs, outputs)
        reconstructed = model_dec.predict(encoding)
        img = np.reshape(reconstructed, (32, 32))
        return img