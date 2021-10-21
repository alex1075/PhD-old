import numpy as np
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, BatchNormalization, Activation, ZeroPadding2D, LeakyReLU, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import glob
import cv2
import matplotlib.pyplot as plt
import imgaug as ia
import imgaug.augmenters as iaa
from imageTools import *

# Generate the ML model for automatic subject detection 
class cycleGAN():

    def __init__(self, N_channels = 1, input_width = 256, input_height = 256, dataset_name = None):

        self.N_channels = N_channels
        self.input_width = input_width
        self.input_height = input_height
        self.dataset_name = dataset_name
        self.image_shape = (self.input_width, self.input_height, self.N_channels)

        self.image_poolA = list()
        self.image_poolB = list()

        self.dataset_A = glob.glob("Data/*.jpg")
        self.dataset_B = glob.glob("Labels/*.png")
        
        self.directory = ''

        self.filtersize = 128

        self.g_model_AB = self.define_generator(self.filtersize)
        self.g_model_BA = self.define_generator(self.filtersize)
        self.d_model_A = self.define_discriminator(self.filtersize)
        self.d_model_B = self.define_discriminator(self.filtersize)

        self.c_model_AB = self.composite_model(self.g_model_AB, self.d_model_B, self.g_model_BA)
        self.c_model_BA = self.composite_model(self.g_model_BA, self.d_model_A, self.g_model_AB)

    # Define Discriminator
    def define_discriminator(self, n_filt):
        # weight initialisation
        init = RandomNormal(stddev=0.02, seed=1)
        input_img = Input(shape=self.image_shape)

        #c1
        c1 = Conv2D(n_filt, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(input_img)
        c1 = LeakyReLU(alpha=0.2)(c1)
        #c2
        c2 = Conv2D(n_filt*2, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(c1)
        c2 = InstanceNormalization(axis=-1)(c2)
        c2 = LeakyReLU(alpha=0.2)(c2)
        #c3
        c3 = Conv2D(n_filt*4, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(c2)
        c3 = InstanceNormalization(axis=-1)(c3)
        c3 = LeakyReLU(alpha=0.2)(c3)
        #c4
        c4 = Conv2D(n_filt*8, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(c3)
        c4 = InstanceNormalization(axis=-1)(c4)
        c4 = LeakyReLU(alpha=0.2)(c4)
        #c5
        c5 = Conv2D(n_filt*8, (4, 4), padding='same', kernel_initializer=init)(c4)
        c5 = InstanceNormalization(axis=-1)(c5)
        c5 = LeakyReLU(alpha=0.2)(c5)
        # Patch Output
        patch_out = Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(c5)
        # Define model
        model = Model(input_img, patch_out)
        model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])

        return model

    def define_generator(self, n_filt, n_resnet_layers=6):
        #weight initialisation
        init = RandomNormal(stddev=0.02, seed=1)
        input_img = Input(shape=self.image_shape)
        # c1
        c1 = Conv2D(n_filt, (7, 7), padding='same', kernel_initializer=init)(input_img)
        c1 = InstanceNormalization(axis=-1)(c1)
        c1 = Activation('relu')(c1)
        # c2
        c2 = Conv2D(n_filt*2, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(c1)
        c2 = InstanceNormalization(axis=-1)(c2)
        c2 = Activation('relu')(c2)
        # c3
        c3 = Conv2D(n_filt*4, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(c2)
        c3 = InstanceNormalization(axis=-1)(c3)
        r = Activation('relu')(c3)
        # ResNet Blocks
        for _ in range(n_resnet_layers):
            r = self.resnet_block(n_filt*4, r)
        #u1
        u1 = Conv2DTranspose(n_filt*2, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(r)
        u1 = InstanceNormalization(axis=-1)(u1)
        u1 = Activation('relu')(u1)
        #u2
        u2 = Conv2DTranspose(n_filt, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(u1)
        u2 = InstanceNormalization(axis=-1)(u2)
        u2 = Activation('relu')(u2)
        #u3
        u3 = Conv2D(self.N_channels, (7, 7), padding='same', kernel_initializer=init)(u2)
        u3 = InstanceNormalization(axis=-1)(u3)
        output = Activation('tanh')(u3)

        model = Model(input_img, output)
        return model

    def composite_model(self, g_model1, d_model, g_model2):
        g_model1.trainable = True
        d_model.trainable = False
        g_model2.trainable = False

        # Discriminator Element
        input_gen = Input(shape=self.image_shape)
        g_model1_output = g_model1(input_gen)
        d_model_output = d_model(g_model1_output)
        # Identity Element
        input_id = Input(shape=self.image_shape)
        output_id = g_model1(input_id)
        # Forward cycle
        output_f = g_model2(g_model1_output)
        # Backward cycle
        g_model2_output = g_model2(input_id)
        output_b = g_model1(g_model2_output)

        model = Model([input_gen, input_id], [d_model_output, output_id, output_f, output_b])
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=opt)

        return model

    def resnet_block(self, n_filt, input_layer):
        #weight initialisation
        init = RandomNormal(stddev=0.2, seed=1)
        
        r = Conv2D(n_filt, (3, 3), padding='same', kernel_initializer=init)(input_layer)
        r = InstanceNormalization(axis=-1)(r)
        r = Activation('relu')(r)

        r = Conv2D(n_filt, (3, 3), padding='same', kernel_initializer=init)(r)
        r = InstanceNormalization(axis=-1)(r)

        return Concatenate()([r, input_layer])
 
   
    #Call this function to generate circles 
    def getRandomCircle(self):
        randomImg = cv2.imread(self.dataset_B[np.random.randint(0, len(self.dataset_B))], -1)
        normalized_cropped = self.normaliseImg(self.getRandomCrop(randomImg, 256, 256))
        return normalized_cropped

   
    def generate_real_samples(self, n_samples, patch_shape):
        
        X = np.zeros((n_samples, self.input_width, self.input_height, self.N_channels))

        # img = cv2.imread(self.dataset_A[np.random.randint(0, len(self.dataset_A))], -1)
        for i in range(n_samples):
            img = cv2.imread(self.dataset_A[np.random.randint(0, len(self.dataset_A))],-1)
            # print(img.shape)
            X[i] = self.normaliseImg(self.getRandomCrop(img, 256, 256))
            print(img.shape)
            # img = self.imageBrightnessDecrease(img, np.random.uniform(9.0, 13.0))
            # print(img.shape)
            # X[i] = self.imageContrastIncrease(img, intensity=np.random.uniform(1.1,1.25))
             
        Y = np.ones((n_samples, patch_shape, patch_shape, 1)) # Labels for real images
        

        return X, Y

    
    def generate_mask_samples(self, n_samples, patch_shape):
        
        X = np.zeros((n_samples, self.input_width, self.input_height, self.N_channels))
        
        for i in range(n_samples):
            X[i] = self.normaliseImg(self.getRandomCircle())
           
        Y = np.ones((n_samples, patch_shape, patch_shape, 1))
        
        return X, Y

    def generate_fake_samples(self, g_model, dataset, patch_shape):
        # Generate fake images
        X = g_model.predict(dataset)

        # These are fake images, so are represented as 0s
        Y = np.zeros((len(X), patch_shape, patch_shape, 1))

        return X, Y

    def update_image_pool(self, img_pool, images, max_size=50):
        selected = list()

        for image in images:
            if len(img_pool) < max_size:
                # Add images to the pool
                img_pool.append(image)
                selected.append(image)
            elif np.random.uniform(0, 1, 1) < 0.5:
                # If pool full, either use a new image
                selected.append(image)
            else:
                # Or replace an existing image and use replacement
                ix = np.random.randint(0, len(img_pool))
                selected.append(img_pool[ix])
                img_pool[ix] = image

        return np.array(selected)

    def setDatasetA_path(self, path):
        filenames = np.array(glob(path))

        self.dataset_A = filenames

    def setDatasetB_path(self, path):
        filenames = np.array(glob(path))

        self.dataset_B = filenames

    def testCycleGAN(self, epoch):
        plt.figure(figsize=(40, 20))
        for i in range(5):
            X_real_A, _ = self.generate_real_samples(1, self.d_model_A.output_shape[1])
            X_real_B, _ = self.generate_mask_samples(1, self.d_model_A.output_shape[1])

            pred_rAfB = self.g_model_AB.predict(X_real_A)
            pred_rBfA = self.g_model_BA.predict(X_real_B)
            pred_fArB = self.g_model_AB.predict(pred_rBfA)
            pred_fBaA = self.g_model_BA.predict(pred_rAfB)

            X_real_A = self.normaliseImgBack(X_real_A)
            X_real_B = self.normaliseImgBack(X_real_B)
            pred_rAfB = self.normaliseImgBack(pred_rAfB)
            pred_rBfA = self.normaliseImgBack(pred_rBfA)
            pred_fArB = self.normaliseImgBack(pred_fArB)
            pred_fBaA = self.normaliseImgBack(pred_fBaA)


            plt.subplot(6, 10, i+1)
            plt.imshow(np.squeeze(X_real_A), cmap="gray")
            plt.axis(False)
            plt.subplot(6, 10, i+1+10)
            plt.imshow(np.squeeze(X_real_B), cmap="gray")
            plt.axis(False)
            plt.subplot(6, 10, i+1+20)
            plt.imshow(np.squeeze(pred_rBfA), cmap="gray")
            plt.axis(False)
            plt.subplot(6, 10, i+1+30)
            plt.imshow(np.squeeze(pred_fArB), cmap="gray")
            plt.axis(False)
            # plt.subplot(6, 10, i+1+40)
            # plt.imshow(np.squeeze(pred_rBfA), cmap="gray")
            # plt.axis(False)
            # plt.subplot(6, 10, i+1+50)
            # plt.imshow(np.squeeze(pred_fArB), cmap="gray")
            # plt.axis(False)

            # plt.subplot(6, 10, i+1)
            # plt.imshow(np.squeeze(X_real_A), cmap="gray")
            # plt.axis(False)
            # plt.subplot(6, 10, i+1+10)
            # plt.imshow(np.squeeze(pred_rAfB), cmap="gray")
            # plt.axis(False)
            # plt.subplot(6, 10, i+1+20)
            # plt.imshow(np.squeeze(pred_fBaA), cmap="gray")
            # plt.axis(False)
            # plt.subplot(6, 10, i+1+30)
            # plt.imshow(np.squeeze(X_real_B), cmap="gray")
            # plt.axis(False)
            # plt.subplot(6, 10, i+1+40)
            # plt.imshow(np.squeeze(pred_rBfA), cmap="gray")
            # plt.axis(False)
            # plt.subplot(6, 10, i+1+50)
            # plt.imshow(np.squeeze(pred_fArB), cmap="gray")
            # plt.axis(False)
        filename = 'epoch_%0.6d.png' % (epoch)
        plt.tight_layout()
        plt.savefig(self.directory + filename)
        plt.close()
        
    def saveModels(self, epoch):
        self.g_model_AB.save(self.directory + 'g_model_AB.h5')
        self.g_model_BA.save(self.directory + 'g_model_BA.h5')
        self.d_model_A.save(self.directory + 'd_model_A.h5')
        self.d_model_B.save(self.directory + 'd_model_B.h5')
        #self.c_model_AB.save('c_model_AB_epoch_%0.6d.h5' % (epoch))
        #self.c_model_BA.save('c_model_BA_epoch_%0.6d.h5' % (epoch))

    def train(self, n_epochs, n_batch):
        
        n_patch = self.d_model_A.output_shape[1]
        #n_steps = int(len(self.dataset_A) / n_batch)
        n_steps = 200

        n = 1

        for i in range(n_epochs):
            for j in range(n_steps):
                X_real_A, Y_real_A = self.generate_real_samples(n_batch, self.d_model_A.output_shape[1]) #particle images --- changed this 
                X_real_B, Y_real_B = self.generate_mask_samples(n_batch, self.d_model_A.output_shape[1]) #masks -- changed this
                
                # Generate fake samples
                X_fake_A, Y_fake_A = self.generate_fake_samples(self.g_model_BA, X_real_B, n_patch) #fake particle images (generator)
                X_fake_B, Y_fake_B = self.generate_fake_samples(self.g_model_AB, X_real_A, n_patch) #fake masks (generator)

                # Update fakes from image pool
                X_fake_A = self.update_image_pool(self.image_poolA, X_fake_A)
                X_fake_B = self.update_image_pool(self.image_poolB, X_fake_B)

                # Update generator B->A via adversarial and cycle loss
                g_loss2, _, _, _, _ = self.c_model_BA.train_on_batch([X_real_B, X_real_A], [Y_real_A, X_real_A, X_real_B, X_real_A])
                dA_loss1 = self.d_model_A.train_on_batch(X_real_A, Y_real_A)
                dA_loss2 = self.d_model_A.train_on_batch(X_fake_A, Y_fake_A)
                g_loss1, _, _, _, _ = self.c_model_AB.train_on_batch([X_real_A, X_real_B], [Y_real_B, X_real_B, X_real_A, X_real_B])
                dB_loss1 = self.d_model_B.train_on_batch(X_real_B, Y_real_B)
                dB_loss2 = self.d_model_B.train_on_batch(X_fake_B, Y_fake_B)

                print('>%d / %d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]' % (n, n_epochs*n_steps, dA_loss1, dA_loss2, dB_loss1, dB_loss2, g_loss1, g_loss2))

                if 'nan' in [str(dA_loss1),  str(dA_loss2), str(dB_loss1),  str(dB_loss2)]:
                    print(f"dA_loss1 = {dA_loss1}, dA_loss2 = {dA_loss2}, dB_loss1 = {dB_loss1}, dB_loss2 = {dB_loss2}")
                    self.testCycleGAN(n)
                    self.testCycleGAN(n-1)
                    self.testCycleGAN(n-2)
                    self.testCycleGAN(n-3)
                    print("Dividing by 0 somewhere")
                    cont = input('do you want to continue training?')
                    if cont == 'yes':
                        continue
                    else: 
                        break

                if np.mod(n, int((n_epochs*n_steps)/200)) == 0:
                    self.testCycleGAN(n)
                    self.saveModels(n)
            
                n += 1