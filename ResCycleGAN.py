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
import keras_contrib
from keras import backend as K


class CycleGAN():

    def __init__(self):
        # Input shape
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.var_img_shape = (None, None, self.channels)

        # Configure data loader
        self.modelName = '/ResCycleGAN'
        self.dataset_name = 'A_to_B'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols),
                                      is_zeroMean=False)


        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 32
        self.df = 64

        # Loss weights
        self.lambda_cycle = 10.0  # Cycle-consistency loss
        self.lambda_id = 1.0      # Identity loss (Default value is 1, Use [0.1, 1.0])

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminators
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()

        print(' ')
        print('* '*30 + 'Discriminators Network' + ' *'*30)
        self.d_A.summary()
        print(' ')

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

        print(' ')
        print('* '*30 + 'Generators Network' + ' *'*30)
        self.g_AB.summary()
        print(' ')

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
        self.combined.compile(loss=['mse', 'mse', self.my_Iloss, self.my_Iloss, self.my_loss, self.my_loss],
                                    loss_weights=[1, 1, self.lambda_id, self.lambda_id, \
                                                  self.lambda_cycle, self.lambda_cycle],
                                    optimizer=optimizer)

        print(' ')
        print('* '*30 + 'Overall Network' + ' *'*30)
        self.combined.summary()
        print(' ')


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


    def train(self, startEpochs, epochs, batch_size=128, save_interval=50):

        half_batch = int(batch_size / 2)

        start_time = datetime.datetime.now()

        for epoch in range(startEpochs, epochs):

            # ----------------------
            #  Train Discriminators
            # ----------------------

            imgs_A = self.data_loader.load_data(domain="A", batch_size=half_batch)
            imgs_B = self.data_loader.load_data(domain="B", batch_size=half_batch)

            # Translate images to opposite domain
            fake_B = self.g_AB.predict(imgs_A)
            fake_A = self.g_BA.predict(imgs_B)

            valid = np.ones((half_batch,) + self.disc_patch)
            fake = np.zeros((half_batch,) + self.disc_patch)

            # Train the discriminators (original images = real / translated = Fake)
            dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
            dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
            dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

            dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
            dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
            dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

            # Total disciminator loss
            d_loss = 0.5 * np.add(dA_loss, dB_loss)


            # ------------------
            #  Train Generators
            # ------------------

            # Sample a batch of images from both domains
            imgs_A = self.data_loader.load_data(domain="A", batch_size=batch_size)
            imgs_B = self.data_loader.load_data(domain="B", batch_size=batch_size)

            # The generators want the discriminators to label the translated images as real
            valid = np.ones((batch_size,) + self.disc_patch)

            # Train the generators
            g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, valid, imgs_A, imgs_B, imgs_A, imgs_B])

            elapsed_time = datetime.datetime.now() - start_time

            if epoch == 0:
                print(np.size(d_loss))
                print(np.size(g_loss))


            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                # Plot the progress
                print ('* '*30)
                print ("%d time: %s" % (epoch, elapsed_time))
                print ("d_loss: %f,  g_loss: %f" % (d_loss[0], g_loss[0]))
                print ("dA_loss_real: %f,  dA_loss_fake: %f" % (dA_loss_real[0], dA_loss_fake[0]))
                print ("dB_loss_real: %f,  dB_loss_fake: %f" % (dB_loss_real[0], dB_loss_fake[0]))
                print (' ')
                self.save_imgs(epoch)

            if epoch % (10000) == 0:
                saveWtPath = "./weights/"+ self.dataset_name + self.modelName + '/CL_' + str(self.lambda_cycle) + '_IL_' + str(self.lambda_id) + '/'
                if not os.path.exists(saveWtPath):
                    os.makedirs(saveWtPath)
                self.d_A.save_weights(saveWtPath + str(epoch) + '_d_A.hdf5')
                self.d_B.save_weights(saveWtPath + str(epoch) + '_d_B.hdf5')
                self.g_AB.save_weights(saveWtPath + str(epoch) + '_d_AB.hdf5')
                self.g_BA.save_weights(saveWtPath + str(epoch) + '_d_BA.hdf5')


    def save_imgs(self, epoch):
        savePath = "./images/"+ self.dataset_name + self.modelName + '/CL_' + str(self.lambda_cycle) + '_IL_' + str(self.lambda_id) + '/'
        if not os.path.exists(savePath):
            os.makedirs(savePath)

        # Demo (for GIF)
        path = glob('./datasets/%s/%s/*' % (self.dataset_name, 'testA'))
        print(np.random.choice(path, size=1))
        imgs_A = self.data_loader.load_full_img(np.random.choice(path, size=1)[0])

        path = glob('./datasets/%s/%s/*' % (self.dataset_name, 'testB'))
        imgs_B = self.data_loader.load_full_img(np.random.choice(path, size=1)[0])

        # Translate images to the other domain
        fake_B = self.g_AB.predict(imgs_A)
        fake_A = self.g_BA.predict(imgs_B)
        # Translate back to original domain
        reconstr_A = self.g_BA.predict(fake_B)
        reconstr_B = self.g_AB.predict(fake_A)

        resName = ['Original', 'Translated', 'Reconstructed']

        path = savePath + "%d_A_%s.png" % (epoch, resName[0])
        self.data_loader.imwrite(path, imgs_A)

        path = savePath + "%d_A_%s.png" % (epoch, resName[1])
        self.data_loader.imwrite(path, fake_B)

        path = savePath + "%d_A_%s.png" % (epoch, resName[2])
        self.data_loader.imwrite(path, reconstr_A)

        path = savePath + "%d_B_%s.png" % (epoch, resName[0])
        self.data_loader.imwrite(path, imgs_B)

        path = savePath + "%d_B_%s.png" % (epoch, resName[1])
        self.data_loader.imwrite(path, fake_A)

        path = savePath + "%d_B_%s.png" % (epoch, resName[2])
        self.data_loader.imwrite(path, reconstr_B)



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


    def loss_MS_SSIM(self, y_true, y_pred):

      y_true = K.clip(y_true, K.epsilon(), 1)
      y_pred = K.clip(y_pred, K.epsilon(), 1)

      # expected net output is of shape [batch_size, row, col, image_channels]
      # We need to shuffle this to [Batch_size, image_channels, row, col]
      y_true = y_true.dimshuffle([0, 3, 1, 2])
      y_pred = y_pred.dimshuffle([0, 3, 1, 2])

      c1 = 0.01 ** 2
      c2 = 0.03 ** 2
      ssim = 1.0

      for i in range(0,3):
        patches_true = K.T.nnet.neighbours.images2neibs(y_true, [8, 8], [4, 4])
        patches_pred = K.T.nnet.neighbours.images2neibs(y_pred, [8, 8], [4, 4])

        mx = K.mean(patches_true, axis=-1)
        my = K.mean(patches_pred, axis=-1)
        varx = K.var(patches_true, axis=-1)
        vary = K.var(patches_pred, axis=-1)
        covxy = K.mean(patches_true*patches_pred, axis=-1) - mx*my

        if i == 0:
          ssimLn = (2 * mx * my + c1)
          ssimLd = (mx ** 2 + my ** 2 + c1)
          ssimLn /= K.clip(ssimLd, K.epsilon(), np.inf)
          ssim = K.mean(K.pow(ssimLn, 3))

        ssimCn = (2 * covxy + c2)
        ssimCd = (vary + varx + c2)
        ssimCn /= K.clip(ssimCd, K.epsilon(), np.inf)

        ssim *= K.mean(ssimCn)

        y_true = K.pool2d(y_true, (2,2),(2,2), data_format='channels_first', pool_mode='avg')
        y_pred = K.pool2d(y_pred, (2,2),(2,2), data_format='channels_first', pool_mode='avg')

      return (1.0 - ssim)

    def my_loss(self, y_true, y_pred):
        beta = 0.80
        ms_ssim_loss = self.loss_MS_SSIM(y_true, y_pred)
        l1_loss = K.mean(K.abs(y_pred - y_true))

        return beta * ms_ssim_loss + (1-beta) * l1_loss

    def my_Iloss(self, y_true, y_pred):
        beta = 1.0
        ms_ssim_loss = self.loss_wMS_SSIM(y_true, y_pred)
        l1_loss = K.mean(K.abs(y_pred - y_true))

        return beta * ms_ssim_loss + (1-beta) * l1_loss

    def loss_wMS_SSIM(self, y_true, y_pred):

      y_true = K.clip(y_true, K.epsilon(), 1)
      y_pred = K.clip(y_pred, K.epsilon(), 1)

      # expected net output is of shape [batch_size, row, col, image_channels]
      # We need to shuffle this to [Batch_size, image_channels, row, col]
      y_true = y_true.dimshuffle([0, 3, 1, 2])
      y_pred = y_pred.dimshuffle([0, 3, 1, 2])

      c1 = 0.01 ** 2
      c2 = 0.03 ** 2
      ssim = 1.0

      alpha = 1.0
      beta = 1.0
      gamma = 1.0

      for i in range(0,3):
        patches_true = K.T.nnet.neighbours.images2neibs(y_true, [8, 8], [4, 4])
        patches_pred = K.T.nnet.neighbours.images2neibs(y_pred, [8, 8], [4, 4])

        mx = K.mean(patches_true, axis=-1)
        my = K.mean(patches_pred, axis=-1)
        varx = K.var(patches_true, axis=-1)
        vary = K.var(patches_pred, axis=-1)
        covxy = K.mean(patches_true*patches_pred, axis=-1) - mx*my

        if i == 0:
            ssimLn = (2 * mx * my + c1)
            ssimLd = (mx**2 + my**2 + c1)
            ssimLn /= K.clip(ssimLd, K.epsilon(), np.inf)
            ssim = K.mean(ssimLn**(alpha * 3))

        ssimCn = (2 * K.sqrt(varx * vary  + K.epsilon()) + c2)
        ssimCd = (varx + vary + c2)
        ssimCn /= K.clip(ssimCd, K.epsilon(), np.inf)

        ssimSn = (covxy + c2/2)
        ssimSd = (K.sqrt(varx * vary + K.epsilon()) + c2/2)
        ssimSn /= K.clip(ssimSd, K.epsilon(), np.inf)

        ssim *= K.mean((ssimCn**beta) * (ssimSn**gamma))

        y_true = K.pool2d(y_true, (2,2),(2,2), data_format='channels_first', pool_mode='avg')
        y_pred = K.pool2d(y_pred, (2,2),(2,2), data_format='channels_first', pool_mode='avg')

      return (1.0 - ssim)


if __name__ == '__main__':
    gan = CycleGAN()
    gan.load_weights(modelNo = 0)
    gan.train(startEpochs = 0, epochs=600000, batch_size=2, save_interval=10000)
