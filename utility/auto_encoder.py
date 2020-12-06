from utility.utils import *
from constants import *


class Autoencoder:

    def encoder(self, input_img):
        conv1 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=(2, 2), activation='relu', padding='same')(
            input_img)  # 100 X 100 X 64
        pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv1)  # 50 X 50 X 64
        conv2 = tf.keras.layers.Conv2D(192, kernel_size=3, strides=(2, 2), activation='relu', padding='same')(
            pool1)  # 50 X 50 X 192
        pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv2)  # 25 X 25 X 192
        conv3 = tf.keras.layers.Conv2D(128, kernel_size=1, strides=(2, 2), activation='relu', padding='same')(
            pool2)  # 25 X 25 X 128
        # conv4_flat = tf.keras.layers.Flatten()(conv3)  # 14 X 2584
        # lin_1 = tf.keras.layers.Dense(3584, activation='relu')(conv4_flat)
        # lin_2 = tf.keras.layers.Dense(512, activation='relu')(lin_1)
        return conv3  # This will be our "z". The encoded vector

    def decoder(self, conv4):
        rev_pool3 = tf.keras.layers.UpSampling2D((2, 2))(conv4)
        rev_con3 = tf.keras.layers.Conv2DTranspose(192, kernel_size=3, strides=(2, 2), activation='relu',
                                                   padding='valid')(rev_pool3)
        rev_conv4 = tf.keras.layers.Conv2DTranspose(192, kernel_size=2, strides=(2, 2), activation='relu',
                                                    padding='same')(rev_pool3)
        rev_conv2 = tf.keras.layers.Conv2DTranspose(64, kernel_size=3, strides=(2, 2), activation='relu',
                                                    padding='valid')(rev_conv4)
        rev_pool1 = tf.keras.layers.UpSampling2D((2, 2))(rev_conv2)  # 50 X 50 X 64
        rev_pool2 = tf.keras.layers.UpSampling2D((2, 2))(rev_pool1)  # 50 X 50 X 64
        deconv_1 = tf.keras.layers.Conv2D(3, kernel_size=1, strides=(1, 1), activation='relu', padding='valid')(
            rev_pool2)
        return deconv_1

    def autoencoder(self, input_img):
        encoder_output = self.encoder(input_img)
        decoder_output = self.decoder(encoder_output)
        return decoder_output

    def initiate_model(self):
        img_width, img_height = 100, 100
        input_img = tf.keras.layers.Input(shape=(img_width, img_height, 3))
        autoencoder_cnn = tf.keras.models.Model(input_img, self.autoencoder(input_img))

        autoencoder_cnn.load_weights(AUONCODER_MODEL, by_name=True)
        autoencoder_cnn._layers.pop(-1)
        autoencoder_cnn._layers.pop(-1)
        autoencoder_cnn._layers.pop(-1)
        autoencoder_cnn._layers.pop(-1)
        autoencoder_cnn._layers.pop(-1)
        autoencoder_cnn._layers.pop(-1)
        autoencoder_cnn.summary()
        return autoencoder_cnn