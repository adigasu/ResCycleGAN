from __future__ import print_function, division

import os
from glob import glob
import numpy as np

from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dropout, Concatenate, Multiply
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
from data_loader import DataLoader


class CycleGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.var_img_shape = (None, None, self.channels)

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 32
        self.df = 64

        # Configure data loader
        self.modelName = '/ResCycleGAN'

        self.dataset_name = 'A_to_B'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols),
                                      is_zeroMean=False)
                                      # is_zeroMean=False)
        # self.is_res = False
        self.is_res = True

        self.is_blockwise = False

        # Loss weights
        self.lambda_cycle = 10.0  # Cycle-consistency loss
        self.lambda_id = 1.0      # Identity loss

        optimizer = Adam(0.001, 0.9)

        # Build and compile the discriminators
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()

        # print(' ')
        # print('* '*30 + 'Discriminators Network' + ' *'*30)
        # self.d_A.summary()
        # print(' ')

        self.d_A.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.d_B.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generators
        self.g_AB = self.build_generator()
        self.g_BA = self.build_generator()
        self.g_AB.compile(loss='binary_crossentropy', optimizer=optimizer)
        self.g_BA.compile(loss='binary_crossentropy', optimizer=optimizer)

        # print(' ')
        # print('* '*30 + 'Generators Network' + ' *'*30)
        # self.g_AB.summary()
        # print(' ')

        # Input images from both domains
        img_A = Input(shape=self.var_img_shape)
        img_B = Input(shape=self.var_img_shape)

        # Translate images to the other domain
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)
        # Translate images back to original domain
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)

        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        self.combined = Model([img_A, img_B], [valid_A, valid_B, fake_B, fake_A, \
                                               reconstr_A, reconstr_B])
        self.combined.compile(loss=['mse', 'mse', 'mae', 'mae', 'mae', 'mae'],
                                    loss_weights=[1, 1, self.lambda_id, self.lambda_id, \
                                                  self.lambda_cycle, self.lambda_cycle],
                                    optimizer=optimizer)

        # print(' ')
        # print('* '*30 + 'Overall Network' + ' *'*30)
        # self.combined.summary()
        # print(' ')

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = InstanceNormalization()(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = InstanceNormalization()(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.var_img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)

        # Upsampling
        u1 = deconv2d(d4, d3, self.gf*4)
        u2 = deconv2d(u1, d2, self.gf*2)
        u3 = deconv2d(u2, d1, self.gf)

        u4 = UpSampling2D(size=2)(u3)

        if not self.is_res:
            output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)
        else:
            out_cor = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='relu')(u4)
            output_img = Multiply()([d0, out_cor])

        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, normalization=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = InstanceNormalization()(d)
            return d

        img = Input(shape=self.var_img_shape)

        d1 = d_layer(img, self.df, normalization=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model(img, validity)

    def test(self, modelNo, batch_size):

        start_time = datetime.datetime.now()

        # ------------------
        #  Load weights
        # ------------------
        self.load_weights(modelNo)

        # ------------------
        #  test Generators
        # ------------------

        self.save_imgs(modelNo, batch_size)
        elapsed_time = datetime.datetime.now() - start_time

        # Plot the progress
        print ('* '*30)
        print ("Elaspsed time: %s" % (elapsed_time))
        print ('* '*30)

    def save_imgs(self, modelNo, batch_size):
        savePath = "./images/"+ self.dataset_name + self.modelName + '/CL_' + str(self.lambda_cycle) + '_IL_' + str(self.lambda_id) + '/' + str(modelNo) + '/'
        if not os.path.exists(savePath):
            os.makedirs(savePath)

        path1 = glob('./datasets/%s/%s/*.*g' % (self.dataset_name, 'testA'))
        #path2 = glob('./datasets/%s/%s/*' % (self.dataset_name, 'topcon'))

        # batch_images1 = np.random.choice(path1, size=batch_size)
        # batch_images2 = np.random.choice(path2, size=batch_size)
        batch_images1 = path1
        # batch_images2 = path1

        k = 0
        for img_path in batch_images1:
            # print()
            # print ('- '*30)
            imgs_A = self.data_loader.load_full_img(batch_images1[k], blockwise = self.is_blockwise, is_ext = True)
            # imgs_B = self.data_loader.load_full_img(batch_images2[k], blockwise = self.is_blockwise)

            # Translate images to the other domain
            fake_B = self.g_AB.predict(imgs_A, batch_size=1)
            # fake_A = self.g_BA.predict(imgs_B, batch_size=1)
            # Translate back to original domain
            # reconstr_A = self.g_BA.predict(fake_B, batch_size=1)
            # reconstr_B = self.g_AB.predict(fake_A, batch_size=1)

            resName = ['Original', 'Translated', 'Reconstructed']

            #path = savePath + "%d_A_%s.png" % (k, resName[0])
            #self.data_loader.imwrite(path, imgs_A, blockwise = self.is_blockwise)

            # path = savePath + "%d_A_%s.png" % (k, resName[1])
            path = savePath + os.path.basename(img_path)
            self.data_loader.imwrite(path, fake_B, blockwise = self.is_blockwise)

            # path = savePath + "%d_A_%s.png" % (k, resName[2])
            # self.data_loader.imwrite(path, reconstr_A, blockwise = self.is_blockwise)

            # path = savePath + "%d_B_%s.png" % (k, resName[0])
            # self.data_loader.imwrite(path, imgs_B, blockwise = self.is_blockwise)

            # path = savePath + "%d_B_%s.png" % (k, resName[1])
            # self.data_loader.imwrite(path, fake_A, blockwise = self.is_blockwise)

            # path = savePath + "%d_B_%s.png" % (k, resName[2])
            # self.data_loader.imwrite(path, reconstr_B, blockwise = self.is_blockwise)

            k = k+1

            # print ('- '*30)
            # print()

    def load_weights(self, modelNo):

        start_time = datetime.datetime.now()

        # ------------------
        #  Load weights of Generators and Discriminators
        # ------------------
        if modelNo != 0:
            saveWtPath = "./weights/"+ self.dataset_name + self.modelName + '/CL_' + str(self.lambda_cycle) + '_IL_' + str(self.lambda_id) + '/'
            self.d_A.load_weights(saveWtPath + str(modelNo) + '_d_A.hdf5')
            self.d_B.load_weights(saveWtPath + str(modelNo) + '_d_B.hdf5')
            self.g_AB.load_weights(saveWtPath + str(modelNo) + '_d_AB.hdf5')
            self.g_BA.load_weights(saveWtPath + str(modelNo) + '_d_BA.hdf5')

        elapsed_time = datetime.datetime.now() - start_time

        # Elapsed time
        print ('* '*30)
        print ("Time for loading weights: %s" % (elapsed_time))
        print ('* '*30)


    def my_loss(self, y_true, y_pred):
        # Dummy
        l1_loss = K.mean(K.abs(y_pred - y_true))

        return l1_loss


if __name__ == '__main__':
    gan = CycleGAN()
    gan.test(modelNo=200000, batch_size=20)
