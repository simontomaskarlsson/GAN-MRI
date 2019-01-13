from keras.layers import ZeroPadding2D, BatchNormalization, Input, MaxPooling2D, AveragePooling2D, Conv2D, LeakyReLU, Flatten, Conv2DTranspose, Activation, add, Lambda, GaussianNoise, merge, concatenate, Dropout
from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers.core import Dense, Flatten, Reshape
from keras.models import Model, load_model
from keras.optimizers import Adam, adam
from keras.activations import tanh
from keras.regularizers import l2
import keras.backend as K
from keras.initializers import RandomNormal
import cv2

from tensorflow.contrib.kfac.python.ops import optimizer
from collections import OrderedDict
from time import localtime, strftime
from scipy.misc import imsave, toimage
import numpy as np
import json
import sys
import time
import datetime

sys.path.append('..')
import load_data
import os
import csv

class UNIT():

    def __init__(self, lr = 1e-4, date_time_string_addition=''):
        self.channels = 3 # 1 for grayscale 3 RGB
        weight_decay = 0.0001/2
        # Load data
        nr_A_train_imgs = 1
        nr_B_train_imgs = 1
        nr_A_test_imgs = 1
        nr_B_test_imgs = None
        image_folder = 'dataset-name/'

        # Fetch data during training instead of pre caching all images - might be necessary for large datasets
        self.use_data_generator = False

        if self.use_data_generator:
            print('--- Using dataloader during training ---')
        else:
            print('--- Caching data ---')
        sys.stdout.flush()

        if self.use_data_generator:
            self.data_generator = load_data.load_data(
                self.channels, generator=True, subfolder=image_folder)
            nr_A_train_imgs=2
            nr_B_train_imgs=2

        data = load_data.load_data(self.channels,
                               nr_A_train_imgs=nr_A_train_imgs,
                               nr_B_train_imgs=nr_B_train_imgs,
                               nr_A_test_imgs=nr_A_test_imgs,
                               nr_B_test_imgs=nr_B_test_imgs,
                               subfolder=image_folder)

        self.A_train = data["trainA_images"]
        self.B_train = data["trainB_images"]
        self.A_test = data["testA_images"]
        self.B_test = data["testB_images"]
        self.testA_image_names = data["testA_image_names"]
        self.testB_image_names = data["testB_image_names"]

        self.img_width = self.A_train.shape[2]
        self.img_height = self.A_train.shape[1]
        self.latent_dim = (int(self.img_height / 4), int(self.img_width / 4), 256)
        self.img_shape = (self.img_height, self.img_width, self.channels)
        self.date_time = strftime("%Y%m%d-%H%M%S", localtime()) + date_time_string_addition
        self.learning_rate = lr
        self.beta_1 = 0.5
        self.beta_2 = 0.999
        self.lambda_0 = 10
        self.lambda_1 = 0.1
        self.lambda_2 = 100
        self.lambda_3 = self.lambda_1 # cycle
        self.lambda_4 = self.lambda_2 # cycle

        # Optimizer
        opt = Adam(self.learning_rate, self.beta_1, self.beta_2)
        optStandAdam = Adam()

        # Simple Model
        self.superSimple = self.modelSimple()
        self.superSimple.compile(optimizer=optStandAdam,
                                 loss="mae")

        # Discriminator
        self.discriminatorA = self.modelMultiDiscriminator("discriminatorA")
        self.discriminatorB = self.modelMultiDiscriminator("discriminatorB")

        for layer in self.discriminatorA.layers:
            if hasattr(layer, 'kernel_regularizer'):
                layer.kernel_regularizer= l2(weight_decay)
                layer.bias_regularizer = l2(weight_decay)
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel_initializer = RandomNormal(mean=0.0, stddev=0.02)
                layer.bias_initializer = RandomNormal(mean=0.0, stddev=0.02)

        for layer in self.discriminatorB.layers:
            if hasattr(layer, 'kernel_regularizer'):
                layer.kernel_regularizer= l2(weight_decay)
                layer.bias_regularizer = l2(weight_decay)
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel_initializer = RandomNormal(mean=0.0, stddev=0.02)
                layer.bias_initializer = RandomNormal(mean=0.0, stddev=0.02)

        self.discriminatorA.compile(optimizer=opt,
                                    loss=['binary_crossentropy',
                                          'binary_crossentropy',
                                          'binary_crossentropy'],
                                    loss_weights=[self.lambda_0,
                                                  self.lambda_0,
                                                  self.lambda_0])
        self.discriminatorB.compile(optimizer=opt,
                                    loss=['binary_crossentropy',
                                          'binary_crossentropy',
                                          'binary_crossentropy'],
                                    loss_weights=[self.lambda_0,
                                                  self.lambda_0,
                                                  self.lambda_0])

        # Encoder
        self.encoderA = self.modelEncoder("encoderA")
        self.encoderB = self.modelEncoder("encoderB")
        self.encoderShared = self.modelSharedEncoder("encoderShared")
        self.decoderShared = self.modelSharedDecoder("decoderShared")

        # Generator
        self.generatorA = self.modelGenerator("generatorA")
        self.generatorB = self.modelGenerator("generatorB")

        # Input Encoder Decoder
        imgA = Input(shape=(self.img_shape))
        imgB = Input(shape=(self.img_shape))
        encodedImageA = self.encoderA(imgA)
        encodedImageB = self.encoderB(imgB)

        sharedA = self.encoderShared(encodedImageA)
        sharedB = self.encoderShared(encodedImageB)

        outSharedA = self.decoderShared(sharedA)
        outSharedB = self.decoderShared(sharedB)

        # Input Generator
        outAa = self.generatorA(outSharedA)
        outBa = self.generatorA(outSharedB)

        outAb = self.generatorB(outSharedA)
        outBb = self.generatorB(outSharedB)

        guess_outBa = self.discriminatorA(outBa)
        guess_outAb = self.discriminatorB(outAb)

        # Cycle
        cycle_encodedImageA = self.encoderA(outBa)
        cycle_encodedImageB = self.encoderB(outAb)

        cycle_sharedA = self.encoderShared(cycle_encodedImageA)
        cycle_sharedB = self.encoderShared(cycle_encodedImageB)

        cycle_outSharedA = self.decoderShared(cycle_sharedA)
        cycle_outSharedB = self.decoderShared(cycle_sharedB)

        cycle_Ab_Ba = self.generatorA(cycle_outSharedB)
        cycle_Ba_Ab = self.generatorB(cycle_outSharedA)

        # Train only generators
        self.discriminatorA.trainable = False
        self.discriminatorB.trainable = False

        self.encoderGeneratorModel = Model(inputs=[imgA, imgB],
                              outputs=[sharedA, sharedB,
                                       cycle_sharedA, cycle_sharedB,
                                       outAa, outBb,
                                       cycle_Ab_Ba, cycle_Ba_Ab,
                                       guess_outBa[0], guess_outAb[0],
                                       guess_outBa[1], guess_outAb[1],
                                       guess_outBa[2], guess_outAb[2]])

        for layer in self.encoderGeneratorModel.layers:
            if hasattr(layer, 'kernel_regularizer'):
                layer.kernel_regularizer= l2(weight_decay)
                layer.bias_regularizer = l2(weight_decay)
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel_initializer = RandomNormal(mean=0.0, stddev=0.02)
                layer.bias_initializer = RandomNormal(mean=0.0, stddev=0.02)

        self.encoderGeneratorModel.compile(optimizer=opt,
                              loss=[self.vae_loss_CoGAN, self.vae_loss_CoGAN,
                                    self.vae_loss_CoGAN, self.vae_loss_CoGAN,
                                    'mae', 'mae',
                                    'mae', 'mae',
                                    'binary_crossentropy', 'binary_crossentropy',
                                    'binary_crossentropy', 'binary_crossentropy',
                                    'binary_crossentropy', 'binary_crossentropy'],
                              loss_weights=[self.lambda_1, self.lambda_1,
                                            self.lambda_3, self.lambda_3,
                                            self.lambda_2, self.lambda_2,
                                            self.lambda_4, self.lambda_4,
                                            self.lambda_0, self.lambda_0,
                                            self.lambda_0, self.lambda_0,
                                            self.lambda_0, self.lambda_0])

#===============================================================================
# Decide what to Run
        self.trainFullModel()
        #self.load_model_and_generate_synthetic_images("name_of_saved_model", epoch) # eg. "20180504-140511_test1", 180
        #self.loadAllWeightsToModelsIncludeDisc("name_of_saved_model", epoch)

#===============================================================================
# Architecture functions

    def resblk(self, x0, k):
        # first layer
        x = Conv2D(filters=k, kernel_size=3, strides=1, padding="same")(x0)
        x = BatchNormalization(axis=3, momentum=0.9, epsilon=1e-05, center=True)(x, training=True)
        x = Activation('relu')(x)
        # second layer
        x = Conv2D(filters=k, kernel_size=3, strides=1, padding="same")(x)
        x = BatchNormalization(axis=3, momentum=0.9, epsilon=1e-05, center=True)(x, training=True)
        x = Dropout(0.5)(x, training=True)
        # merge
        x = add([x, x0])

        return x

#===============================================================================
# Loss function from PyTorch implementation from original article

    def vae_loss_CoGAN(self, y_true, y_pred):
        y_pred_2 = K.square(y_pred)
        encoding_loss = K.mean(y_pred_2)
        return encoding_loss

#===============================================================================
# Models

    def modelMultiDiscriminator(self, name):
        x1 = Input(shape=self.img_shape)
        x2 = AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(x1)
        x4 = AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(x2)

        x1_out = self.modelDiscriminator(x1)
        x2_out = self.modelDiscriminator(x2)
        x4_out = self.modelDiscriminator(x4)

        return Model(inputs=x1, outputs=[x1_out, x2_out, x4_out], name=name)

    def modelDiscriminator(self, x):
        # Layer 1
        x = Conv2D(64, kernel_size=3, strides=2, padding='same')(x)
        x = LeakyReLU(alpha=0.01)(x)
        # Layer 2
        x = Conv2D(128, kernel_size=3, strides=2, padding='same')(x)
        x = LeakyReLU(alpha=0.01)(x)
        # Layer 3
        x = Conv2D(256, kernel_size=3, strides=2, padding='same')(x)
        x = LeakyReLU(alpha=0.01)(x)
        # Layer 4
        x = Conv2D(512, kernel_size=3, strides=2, padding='same')(x)
        x = LeakyReLU(alpha=0.01)(x)
        # Layer 5
        x = Conv2D(1, kernel_size=1, strides=1)(x)
        prediction = Activation('sigmoid')(x)

        return prediction

    def modelEncoder(self, name):
        inputImg = Input(shape=self.img_shape)
        # Layer 1
        x = ZeroPadding2D(padding=(3, 3))(inputImg)
        x = Conv2D(64, kernel_size=7, strides=1, padding='valid')(x)
        x = LeakyReLU(alpha=0.01)(x)
        # Layer 2
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(128, kernel_size=3, strides=2, padding='valid')(x)
        x = LeakyReLU(alpha=0.01)(x)
        # Layer 3
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(256, kernel_size=3, strides=2, padding='valid')(x)
        x = LeakyReLU(alpha=0.01)(x)
        # Layer 4: 2 res block
        x = self.resblk(x, 256)
        # Layer 5: 3 res block
        x = self.resblk(x, 256)
        # Layer 6: 3 res block
        z = self.resblk(x, 256)

        return Model(inputs=inputImg, outputs=z, name=name)

    def modelSharedEncoder(self, name):
        input = Input(shape=self.latent_dim)

        x = self.resblk(input, 256)
        z = GaussianNoise(stddev=1)(x, training=True)

        return Model(inputs=input, outputs=z, name=name)

    def modelSharedDecoder(self, name):
        input = Input(shape=self.latent_dim)

        x = self.resblk(input, 256)

        return Model(inputs=input, outputs=x, name=name)

    def modelGenerator(self, name):
        inputImg = Input(shape=self.latent_dim)
        # Layer 1: 1 res block
        x = self.resblk(inputImg, 256)
        # Layer 2: 2 res block
        x = self.resblk(x, 256)
        # Layer 3: 3 res block
        x = self.resblk(x, 256)
        # Layer 4:
        x = Conv2DTranspose(128, kernel_size=3, strides=2, padding='same')(x)
        x = LeakyReLU(alpha=0.01)(x)
        # Layer 5:
        x = Conv2DTranspose(64, kernel_size=3, strides=2, padding='same')(x)
        x = LeakyReLU(alpha=0.01)(x)
        # Layer 6
        x = Conv2DTranspose(self.channels, kernel_size=1, strides=1, padding='valid')(x)
        z = Activation("tanh")(x)

        return Model(inputs=inputImg, outputs=z, name=name)

    def modelSimple(self):
        inputImg = Input(shape=self.img_shape)
        x = Conv2D(256, kernel_size=1, strides=1, padding='same')(inputImg)
        x = Activation('relu')(x)
        prediction = Conv2D(1, kernel_size=5, strides=1, padding='same')(x)
        return Model(input=inputImg, output=prediction)

#===============================================================================
# Training

    def trainFullModel(self, epochs=100, batch_size=1, save_interval=1):
        def run_training_iteration(loop_index, epoch_iterations, imgA, imgB):
            # Flip was not done in article
            # if np.random.rand(1) > 0.5:
            #     imgA = cv2.flip(imgA[0], 1)
            #     imgA = imgA[np.newaxis,:,:,:]
            # if np.random.rand(1) > 0.5:
            #     imgB = cv2.flip(imgB[0], 1)
            #     imgB = imgB[np.newaxis,:,:,:]

            # Generate fake images
            encodedImageA = self.encoderA.predict(imgA)
            encodedImageB = self.encoderB.predict(imgB)

            sharedA = self.encoderShared.predict(encodedImageA)
            sharedB = self.encoderShared.predict(encodedImageB)

            outSharedA = self.decoderShared.predict(sharedA)
            outSharedB = self.decoderShared.predict(sharedB)

            outAa = self.generatorA.predict(outSharedA)
            outBa = self.generatorA.predict(outSharedB)

            outAb = self.generatorB.predict(outSharedA)
            outBb = self.generatorB.predict(outSharedB)

            # Train discriminator
            dA_loss_real = self.discriminatorA.train_on_batch(imgA, real_labels)
            dA_loss_fake = self.discriminatorA.train_on_batch(outBa, synthetic_labels)
            dA_loss = np.add(dA_loss_real, dA_loss_fake)

            dB_loss_real = self.discriminatorB.train_on_batch(imgB, real_labels)
            dB_loss_fake = self.discriminatorB.train_on_batch(outAb, synthetic_labels)
            dB_loss = np.add(dB_loss_real, dB_loss_fake)

            # Train generator
            g_loss = self.encoderGeneratorModel.train_on_batch([imgA, imgB],
                                                  [dummy, dummy,
                                                   dummy, dummy,
                                                   imgA, imgB,
                                                   imgA, imgB,
                                                   real_labels1, real_labels1,
                                                   real_labels2, real_labels2,
                                                   real_labels3, real_labels3])

            # Store training data
            epoch_list.append(epoch)
            loop_index_list.append(loop_index)

            # Discriminator loss
            loss_dA_real_list.append(dA_loss_real[0])
            loss_dA_fake_list.append(dA_loss_fake[0])
            loss_dB_real_list.append(dB_loss_real[0])
            loss_dB_fake_list.append(dB_loss_fake[0])

            dA_sum_loss_list.append(dA_loss[0])
            dB_sum_loss_list.append(dB_loss[0])

            # Generator loss
            loss_gen_list_1.append(g_loss[0])
            loss_gen_list_2.append(g_loss[1])
            loss_gen_list_3.append(g_loss[2])
            loss_gen_list_4.append(g_loss[3])
            loss_gen_list_5.append(g_loss[4])
            loss_gen_list_6.append(g_loss[5])
            loss_gen_list_7.append(g_loss[6])
            loss_gen_list_8.append(g_loss[7])
            loss_gen_list_9.append(g_loss[8])
            loss_gen_list_10.append(g_loss[9])
            loss_gen_list_11.append(g_loss[10])

            print('----------------Epoch-------640x480---------', epoch, '/', epochs - 1)
            print('----------------Loop index-----------', loop_index, '/', epoch_iterations - 1)
            print('Discriminator TOTAL loss: ', dA_loss[0] + dB_loss[0])
            print('Discriminator A loss total: ', dA_loss[0])
            print('Discriminator B loss total: ', dB_loss[0])
            print('Genarator loss total: ', g_loss[0])
            print('----------------Discriminator loss----')
            print('dA_loss_real: ', dA_loss_real[0])
            print('dA_loss_fake: ', dA_loss_fake[0])
            print('dB_loss_real: ', dB_loss_real[0])
            print('dB_loss_fake: ', dB_loss_fake[0])
            print('----------------Generator loss--------')
            print('Shared A: ', g_loss[1])
            print('Shared B: ', g_loss[2])
            print('Cycle shared A: ', g_loss[3])
            print('Cycle shared B: ', g_loss[4])
            print('OutAa MAE: ', g_loss[5])
            print('OutBb MAE: ', g_loss[6])
            print('Cycle_Ab_Ba MAE: ', g_loss[7])
            print('Cycle_Ba_Ab MAE: ', g_loss[8])
            print('guess_outBa: ', g_loss[9])
            print('guess_outAb: ', g_loss[10])
            print('guess_outBa: ', g_loss[11])
            print('guess_outAb: ', g_loss[12])
            print('guess_outBa: ', g_loss[13])
            print('guess_outAb: ', g_loss[14])
            sys.stdout.flush()

            if loop_index % 5 == 0:
                # Save temporary images continously
                self.save_tmp_images(imgA, imgB)
                self.print_ETA(start_time, epoch, epoch_iterations, loop_index)

        A_train = self.A_train
        B_train = self.B_train
        self.history = OrderedDict()
        self.epochs = epochs
        self.batch_size = batch_size

        loss_dA_real_list = []
        loss_dA_fake_list = []
        loss_dB_real_list = []
        loss_dB_fake_list = []
        dA_sum_loss_list = []
        dB_sum_loss_list = []

        loss_gen_list_1 = []
        loss_gen_list_2 = []
        loss_gen_list_3 = []
        loss_gen_list_4 = []
        loss_gen_list_5 = []
        loss_gen_list_6 = []
        loss_gen_list_7 = []
        loss_gen_list_8 = []
        loss_gen_list_9 = []
        loss_gen_list_10 = []
        loss_gen_list_11 = []
        epoch_list = []
        loop_index_list = []

        #dummy = []
        #dummy = shape=self.latent_dim
        #dummy = np.zeros(shape=self.latent_dim)
        #dummy = np.expand_dims(dummy, 0)
        dummy = np.zeros(shape = ((self.batch_size,) + self.latent_dim))

        self.writeMetaDataToJSON()
        self.saveImages('init', 1)
        sys.stdout.flush()

        # Start stopwatch for ETAs
        start_time = time.time()

        label_shape1 = (batch_size,) + self.discriminatorA.output_shape[0][1:]
        label_shape2 = (batch_size,) + self.discriminatorA.output_shape[1][1:]
        label_shape3 = (batch_size,) + self.discriminatorA.output_shape[2][1:]

        real_labels1 = np.ones(label_shape1)
        real_labels2 = np.ones(label_shape2)
        real_labels3 = np.ones(label_shape3)
        synthetic_labels1 = np.zeros(label_shape1)
        synthetic_labels2 = np.zeros(label_shape2)
        synthetic_labels3 = np.zeros(label_shape3)

        real_labels = [real_labels1, real_labels2, real_labels3]
        synthetic_labels = [synthetic_labels1, synthetic_labels2, synthetic_labels3]

        for epoch in range(epochs):
            if self.use_data_generator:
                loop_index = 1
                for images in self.data_generator:
                    imgA = images[0]
                    imgB = images[1]

                    # Run all training steps
                    run_training_iteration(loop_index, self.data_generator.__len__(), imgA, imgB)
                    print("-----------------Loop Index:", loop_index)
                    if loop_index % 20000 == 0: # 20000
                        self.saveCurrentModels(loop_index)
                        self.saveImages(loop_index, 1)
                    elif loop_index >= self.data_generator.__len__():
                        break
                    loop_index += 1

            else:  # Train with all data in cache
                A_train = self.A_train
                B_train = self.B_train
                random_order_A = np.random.randint(len(A_train), size=len(A_train))
                random_order_B = np.random.randint(len(B_train), size=len(B_train))
                epoch_iterations = max(len(random_order_A), len(random_order_B))
                min_nr_imgs = min(len(random_order_A), len(random_order_B))

                for loop_index in range(0, epoch_iterations, batch_size):
                    if loop_index + batch_size >= min_nr_imgs:
                        # If all images soon are used for one domain,
                        # randomly pick from this domain
                        if len(A_train) <= len(B_train):
                            indexes_A = np.random.randint(len(A_train), size=batch_size)
                            indexes_B = random_order_B[loop_index:
                                                       loop_index + batch_size]
                        else:
                            indexes_B = np.random.randint(len(B_train), size=batch_size)
                            indexes_A = random_order_A[loop_index:
                                                       loop_index + batch_size]
                    else:
                        indexes_A = random_order_A[loop_index:
                                                   loop_index + batch_size]
                        indexes_B = random_order_B[loop_index:
                                                   loop_index + batch_size]

                    imgA = A_train[indexes_A]
                    imgB = B_train[indexes_B]

                    run_training_iteration(loop_index, epoch_iterations, imgA, imgB)

            if epoch % 10 == 0:
                self.saveCurrentModels(epoch)

            if epoch % save_interval == 0:
                print('--------Saving images for epoch', epoch, '--------')
                self.saveImages(epoch, 3)

                # Create dictionary of losses and save to file
                self.history = {
                    'loss_dA_real_list': loss_dA_real_list,
                    'loss_dA_fake_list': loss_dA_fake_list,
                    'loss_dB_real_list': loss_dB_real_list,
                    'loss_dB_fake_list': loss_dB_fake_list,

                    'dA_sum_loss_list': dA_sum_loss_list,
                    'dB_sum_loss_list': dB_sum_loss_list,

                    'loss_gen_list_1': loss_gen_list_1,
                    'loss_gen_list_2': loss_gen_list_2,
                    'loss_gen_list_3': loss_gen_list_3,
                    'loss_gen_list_4': loss_gen_list_4,
                    'loss_gen_list_5': loss_gen_list_5,
                    'loss_gen_list_6': loss_gen_list_6,
                    'loss_gen_list_7': loss_gen_list_7,
                    'loss_gen_list_8': loss_gen_list_8,
                    'loss_gen_list_9': loss_gen_list_9,

                    'loop_index': loop_index_list,
                    'epoch': epoch_list}
                self.writeLossDataToFile()

        self.saveModel(self.discriminatorA, 'discriminatorA', epoch)
        self.saveModel(self.discriminatorB, 'discriminatorB', epoch)
        self.saveModel(self.generatorA, 'generatorA', epoch)
        self.saveModel(self.generatorB, 'generatorB', epoch)
        self.saveModel(self.encoderA, 'encoderA', epoch)
        self.saveModel(self.encoderB, 'encoderB', epoch)
        self.saveModel(self.decoderShared, 'decoderShared', epoch)
        self.saveModel(self.encoderShared, 'encoderShared', epoch)
        sys.stdout.flush()

    def saveCurrentModels(self, epoch):
        self.saveModel(self.discriminatorA, 'discriminatorA', epoch)
        self.saveModel(self.discriminatorB, 'discriminatorB', epoch)
        self.saveModel(self.generatorA, 'generatorA', epoch)
        self.saveModel(self.generatorB, 'generatorB', epoch)
        self.saveModel(self.encoderA, 'encoderA', epoch)
        self.saveModel(self.encoderB, 'encoderB', epoch)
        self.saveModel(self.decoderShared, 'decoderShared', epoch)
        self.saveModel(self.encoderShared, 'encoderShared', epoch)

    def trainSimpleModel(self, epochs=200, batch_size=1, T=1):
        A_train = self.A_train
        B_train = self.B_train

        if T == 1:
            X_train = self.A_train
            Y_train = self.B_train
        else:
            Y_train = self.A_train
            X_train = self.B_train

        self.superSimple.fit(x=X_train, y=Y_train, batch_size=1, epochs=epochs, verbose=1, callbacks=None, validation_split=0.0,
            validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0,
            steps_per_epoch=None, validation_steps=None)

        self.saveImagesSimpleModel(epoch=epochs, num_saved_images=10, T=T)
        self.saveModel(self.superSimple, 'superSimpleModel', epochs)

#===============================================================================
# Save and load Models

    def saveModel(self, model, model_name, epoch):
        directory = os.path.join('saved_model', self.date_time)
        if not os.path.exists(directory):
            os.makedirs(directory)

        model_path_w = 'saved_model/{}/{}_epoch_{}_weights.hdf5'.format(self.date_time, model_name, epoch)
        model.save_weights(model_path_w)
        model_path_m = 'saved_model/{}/{}_epoch_{}_model.json'.format(self.date_time, model_name, epoch)
        model.save_weights(model_path_m)

    def loadAllWeightsToModelsIncludeDisc(self, folder_name, epoch):
        pathEncoderA = 'saved_model/{}/encoderA_epoch_{}_weights.hdf5'.format(folder_name, epoch)
        pathEncoderB = 'saved_model/{}/encoderB_epoch_{}_weights.hdf5'.format(folder_name, epoch)

        pathEncoderShared = 'saved_model/{}/encoderShared_epoch_{}_weights.hdf5'.format(folder_name, epoch)
        pathDecoderShared = 'saved_model/{}/decoderShared_epoch_{}_weights.hdf5'.format(folder_name, epoch)

        pathGeneratorA = 'saved_model/{}/generatorA_epoch_{}_weights.hdf5'.format(folder_name, epoch)
        pathGeneratorB = 'saved_model/{}/generatorB_epoch_{}_weights.hdf5'.format(folder_name, epoch)

        pathDiscriminatorA = 'saved_model/{}/discriminatorA_epoch_{}_weights.hdf5'.format(folder_name, epoch)
        pathDiscriminatorB = 'saved_model/{}/discriminatorB_epoch_{}_weights.hdf5'.format(folder_name, epoch)

        self.encoderA.load_weights(pathEncoderA)
        self.encoderB.load_weights(pathEncoderB)

        self.encoderShared.load_weights(pathEncoderShared)
        self.decoderShared.load_weights(pathDecoderShared)

        self.generatorA.load_weights(pathGeneratorA)
        self.generatorB.load_weights(pathGeneratorB)

        self.discriminatorA.load_weights(pathDiscriminatorA)
        self.discriminatorB.load_weights(pathDiscriminatorB)

    def loadAllWeightsToModels(self, folder_name, epoch):
        pathEncoderA = 'saved_model/{}/encoderA_epoch_{}_weights.hdf5'.format(folder_name, epoch)
        pathEncoderB = 'saved_model/{}/encoderB_epoch_{}_weights.hdf5'.format(folder_name, epoch)

        pathEncoderShared = 'saved_model/{}/encoderShared_epoch_{}_weights.hdf5'.format(folder_name, epoch)
        pathDecoderShared = 'saved_model/{}/decoderShared_epoch_{}_weights.hdf5'.format(folder_name, epoch)

        pathGeneratorA = 'saved_model/{}/generatorA_epoch_{}_weights.hdf5'.format(folder_name, epoch)
        pathGeneratorB = 'saved_model/{}/generatorB_epoch_{}_weights.hdf5'.format(folder_name, epoch)

        self.encoderA.load_weights(pathEncoderA)
        self.encoderB.load_weights(pathEncoderB)

        self.encoderShared.load_weights(pathEncoderShared)
        self.decoderShared.load_weights(pathDecoderShared)

        self.generatorA.load_weights(pathGeneratorA)
        self.generatorB.load_weights(pathGeneratorB)

    def load_model_and_generate_synthetic_images(self, folder_name, epoch):
        self.loadAllWeightsToModels(folder_name, epoch)
        synthetic_images_B = self.predict_A_B(self.A_test)
        synthetic_images_A = self.predict_B_A(self.B_test)

        def save_image(image, name, domain):
            if self.channels == 1:
                image = image[:, :, 0]
            image = np.clip(image/2 + 0.5, 0, 1)

            directory = os.path.join('generate_images_test', folder_name, domain)
            if not os.path.exists(directory):
                os.makedirs(directory)
            directory = os.path.join(directory, name)
            try:
                toimage(image, cmin=0, cmax=1).save(directory)
            except Exception as e:
                print("type error: " + str(e))

        # Test A images
        for i in range(len(synthetic_images_A)):
            # Get the name from the image it was conditioned on
            name = self.testB_image_names[i].strip('.png') + '_synthetic.png'
            synt_A = synthetic_images_A[i]
            save_image(synt_A, name, 'A')

        # Test B images
        for i in range(len(synthetic_images_B)):
            # Get the name from the image it was conditioned on
            name = self.testA_image_names[i].strip('.png') + '_synthetic.png'
            synt_B = synthetic_images_B[i]
            save_image(synt_B, name, 'B')

        print('{} synthetic images have been generated and placed in ./generate_images/synthetic_images'
              .format(len(self.A_test) + len(self.B_test)))

    def predict_A_B(self, imgA):
        encodedImageA = self.encoderA.predict(imgA)
        sharedA = self.encoderShared.predict(encodedImageA)
        outSharedA = self.decoderShared.predict(sharedA)
        outAb = self.generatorB.predict(outSharedA)

        return outAb

    def predict_B_A(self, imgB):
        encodedImageB = self.encoderB.predict(imgB)
        sharedB = self.encoderShared.predict(encodedImageB)
        outSharedB = self.decoderShared.predict(sharedB)
        outBa = self.generatorA.predict(outSharedB)

        return outBa

    def truncateAndSave(self, real_A, real_B, synthetic, reconstructed, epoch, sample, name, filename, tmp=False):
        synthetic = synthetic.clip(min=-1)
        if reconstructed is not None:
            reconstructed = reconstructed.clip(min=-1)
        # Append and save
        if tmp:
             imsave('images/{}/{}.png'.format(
                self.date_time, name), synthetic)
        else:
            if real_A is None and real_B is None:
                imsave('images/{}/{}/{}_synt.png'.format(
                self.date_time, name, filename), synthetic)
            elif real_B is None and reconstructed is None:
                image = np.hstack((real_A, synthetic))
            elif real_A is not None:
                image = np.hstack((real_B, real_A, synthetic, reconstructed))
            else:
                image = np.hstack((real_B, synthetic, reconstructed))
            imsave('images/{}/{}/epoch{}_sample{}.png'.format(
                self.date_time, name, epoch, sample), image)

    def saveImages(self, epoch, num_saved_images=1):

        directory = os.path.join('images', self.date_time)
        if not os.path.exists(os.path.join(directory, 'A')):
            os.makedirs(os.path.join(directory, 'A'))
            os.makedirs(os.path.join(directory, 'B'))

        for i in range(num_saved_images):
            imgA = self.A_test[i]
            imgB = self.B_test[i]

            imgA = np.expand_dims(imgA, axis=0)
            imgB = np.expand_dims(imgB, axis=0)

            # Generate fake images
            encodedImageA = self.encoderA.predict(imgA)
            encodedImageB = self.encoderB.predict(imgB)

            sharedA = self.encoderShared.predict(encodedImageA)
            sharedB = self.encoderShared.predict(encodedImageB)

            outSharedA = self.decoderShared.predict(sharedA)
            outSharedB = self.decoderShared.predict(sharedB)

            outAa = self.generatorA.predict(outSharedA)
            outBa = self.generatorA.predict(outSharedB)

            outAb = self.generatorB.predict(outSharedA)
            outBb = self.generatorB.predict(outSharedB)

            # Cycle
            encodedImageC_A = self.encoderA.predict(outBa)
            encodedImageC_B = self.encoderB.predict(outAb)

            sharedC_A = self.encoderShared.predict(encodedImageC_A)
            sharedC_B = self.encoderShared.predict(encodedImageC_B)

            outSharedC_A = self.decoderShared.predict(sharedC_A)
            outSharedC_B = self.decoderShared.predict(sharedC_B)

            outC_Ba = self.generatorA.predict(outSharedC_B)
            outC_Ab = self.generatorB.predict(outSharedC_A)

            print('')
            if self.channels == 1:
                imgA = imgA[0, :, :, 0]
                outAb0 = outAb[0, :, :, 0]
                outC_Ab = outC_Ab[0, :, :, 0]
                imgB = imgB[0, :, :, 0]
                outBa0 = outBa[0, :, :, 0]
                outC_Ba = outC_Ba[0, :, :, 0]

                self.truncateAndSave(imgA, imgB, outAb0, outC_Ba, epoch, i, 'B', None)
                self.truncateAndSave(imgB, imgA, outBa0, outC_Ab, epoch, i, 'A', None)
            else:
                imgA = imgA[0, :, :, :]
                outAb0 = outAb[0, :, :, :]
                outAa0 = outAa[0, :, :, :]
                imgB = imgB[0, :, :, :]
                outBa0 = outBa[0, :, :, :]
                outBb0 = outBb[0, :, :, :]

                outC_Ab = outC_Ab[0, :, :, :]
                outC_Ba = outC_Ba[0, :, :, :]

                self.truncateAndSave(None, imgA, outAb0, outC_Ba, epoch, i, 'A', None)
                self.truncateAndSave(None, imgB, outBa0, outC_Ab, epoch, i, 'B', None)

    def saveImagesSimpleModel(self, epoch, num_saved_images=1, T=1):
        directory = os.path.join('images', self.date_time)
        if not os.path.exists(os.path.join(directory, 'A')):
            os.makedirs(os.path.join(directory, 'A'))
            os.makedirs(os.path.join(directory, 'B'))

        for i in range(num_saved_images):
            if T == 1:
                img_real = self.A_train[i]
                img_real = np.expand_dims(img_real, axis=0)

            if T == 2:
                img_real = self.B_train[i]
                img_real = np.expand_dims(img_real, axis=0)


            print("Max", np.max(img_real.flatten()))

            print("Min", np.min(img_real.flatten()))

            # Generate fake images
            img_synt = self.superSimple.predict(img_real)
            img_real = img_real[0, :, :, 0]
            img_synt = img_synt[0, :, :, 0]
            self.truncateAndSave(img_real, None, img_synt, None, epoch, i, 'A', i, None)

    def save_tmp_images(self, imgA, imgB):
        try:
            # Generate fake images
            encodedImageA = self.encoderA.predict(imgA)
            encodedImageB = self.encoderB.predict(imgB)

            sharedA = self.encoderShared.predict(encodedImageA)
            sharedB = self.encoderShared.predict(encodedImageB)

            outSharedA = self.decoderShared.predict(sharedA)
            outSharedB = self.decoderShared.predict(sharedB)

            outAa = self.generatorA.predict(outSharedA)
            outBa = self.generatorA.predict(outSharedB)

            outAb = self.generatorB.predict(outSharedA)
            outBb = self.generatorB.predict(outSharedB)

            # Cycle
            encodedImageC_A = self.encoderA.predict(outBa)
            encodedImageC_B = self.encoderB.predict(outAb)

            sharedC_A = self.encoderShared.predict(encodedImageC_A)
            sharedC_B = self.encoderShared.predict(encodedImageC_B)

            outSharedC_A = self.decoderShared.predict(sharedC_A)
            outSharedC_B = self.decoderShared.predict(sharedC_B)

            outC_Ba = self.generatorA.predict(outSharedC_B)
            outC_Ab = self.generatorB.predict(outSharedC_A)

            if self.channels == 1:
                imgA = imgA[0, :, :, 0]
                outAa0 = outAa[0, :, :, 0]
                outAb0 = outAb[0, :, :, 0]
                outC_Ab0 = outC_Ab[0, :, :, 0]
                imgB = imgB[0, :, :, 0]
                outBb0 = outBb[0, :, :, 0]
                outBa0 = outBa[0, :, :, 0]
                outC_Ba0 = outC_Ba[0, :, :, 0]

            else:
                imgA = imgA[0, :, :, :]
                outAa0 = outAa[0, :, :, :]
                outAb0 = outAb[0, :, :, :]
                outC_Ab0 = outC_Ab[0, :, :, :]
                imgB = imgB[0, :, :, :]
                outBb0 = outBb[0, :, :, :]
                outBa0 = outBa[0, :, :, :]
                outC_Ba0 = outC_Ba[0, :, :, :]

            real_images = np.vstack((imgA, imgB))
            recon_images = np.vstack((outAa0, outBb0))
            synthetic_images = np.vstack((outAb0, outBa0))
            recon_cycle_images = np.vstack((outC_Ba0, outC_Ab0))
            image1 = np.hstack((real_images, recon_images))
            image2 = np.hstack((synthetic_images, recon_cycle_images))
            image_tot = np.hstack((image1, image2))

            image_tot_clip = np.clip(image_tot/2 + 0.5, 0, 1)

            np.save('images/{}/{}.npy'.format(self.date_time, 'tmp'), image_tot)
            imsave('images/{}/{}.png'.format(self.date_time, 'tmp'), image_tot_clip)
        except:  # Ignore if file is open
            pass

    def print_ETA(self, start_time, epoch, epoch_iterations, loop_index):
        passed_time = time.time() - start_time
        iterations_so_far = (epoch * epoch_iterations + loop_index) / self.batch_size + 1e-5
        iterations_total = self.epochs * epoch_iterations / self.batch_size
        iterations_left = iterations_total - iterations_so_far
        eta = round(passed_time  / iterations_so_far * iterations_left)
        passed_time_string = str(datetime.timedelta(seconds=round(passed_time)))
        eta_string = str(datetime.timedelta(seconds=eta))
        print('Time passed', passed_time_string, ': ETA in', eta_string)

    def writeLossDataToFile(self):
        keys = sorted(self.history.keys())
        with open('images/{}/loss_output.csv'.format(self.date_time), 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(keys)
            writer.writerows(zip(*[self.history[key] for key in keys]))

    def writeMetaDataToJSON(self):
        directory = os.path.join('images', self.date_time)
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Save meta_data
        data = {}
        data['meta_data'] = []
        data['meta_data'].append({
            'Learning Rate': self.learning_rate,
            'beta 1': self.beta_1,
            'beta 2': self.beta_2,
            'img height': self.img_height,
            'img width': self.img_width,
            'channels': self.channels,
            'epochs': self.epochs,
            'batch size': self.batch_size,
            'number of S1 train examples': len(self.A_train),
            'number of S2 train examples': len(self.B_train),
            'number of S1 test examples': len(self.A_test),
            'number of S2 test examples': len(self.B_test),
        })

        with open('images/{}/meta_data.json'.format(self.date_time), 'w') as outfile:
            json.dump(data, outfile, sort_keys=True)

if __name__ == '__main__':
    np.random.seed(10)
    model = UNIT()
