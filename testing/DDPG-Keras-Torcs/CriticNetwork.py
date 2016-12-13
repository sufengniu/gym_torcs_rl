import numpy as np
import math
from keras.initializations import normal, identity
from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.layers.convolutional import Convolution2D, MaxPooling2D
# from keras.engine.training import collect_trainable_weights
from keras.layers import Dense, Flatten, Input, merge, Lambda, Merge
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600

class CriticNetwork(object):
    def __init__(self, sess, state_size, image_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size

        K.set_session(sess)

        #Now create the model
        self.model, self.action, self.state, self.image = self.create_critic_network(state_size, image_size, action_size)
        self.target_model, self.target_action, self.target_state, self.target_image = self.create_critic_network(state_size, image_size, action_size)
        self.action_grads = tf.gradients(self.model.output, self.action)  #GRADIENTS for policy update
        self.sess.run(tf.initialize_all_variables())

    def gradients(self, states, images, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.image: images,
            self.action: actions
        })[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in xrange(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self, state_size, image_size, action_dim):
        print("Now we build cnn model")
        I = Input(shape=image_size)
        I0 = Convolution2D(32, 8, 8, subsample=(4,4), activation='relu',
            init=lambda shape, name: normal(shape, scale=0.01, name=name),
            border_mode='same',input_shape=image_size)(I)
        I1 = Convolution2D(64, 4, 4, subsample=(2,2), activation='relu',
            init=lambda shape, name: normal(shape, scale=0.01, name=name),
            border_mode='same')(I0)
        I2 = Convolution2D(64, 3, 3, subsample=(1,1), activation='relu',
            init=lambda shape, name: normal(shape, scale=0.01, name=name),
            border_mode='same')(I1)
        I2_5 = Flatten()(I2)
        I3 = Dense(512, activation='relu',
            init=lambda shape, name: normal(shape, scale=0.01, name=name))(I2_5)
        I4 = Dense(HIDDEN2_UNITS, activation='relu',init=lambda shape, name: normal(shape, scale=0.01, name=name))(I3)
        print("Now we build the model")
        S = Input(shape=[state_size])
        A = Input(shape=[action_dim],name='action2')
        w1 = Dense(HIDDEN1_UNITS, activation='relu')(S)
        a1 = Dense(HIDDEN2_UNITS, activation='linear')(A)
        h1 = Dense(HIDDEN2_UNITS, activation='linear')(w1)
        h2 = merge([h1,a1,I4],mode='sum')
        h3 = Dense(HIDDEN2_UNITS, activation='relu')(h2)
        V = Dense(action_dim,activation='linear')(h3)
        model = Model(input=[S,A,I],output=V)
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        return model, A, S, I
