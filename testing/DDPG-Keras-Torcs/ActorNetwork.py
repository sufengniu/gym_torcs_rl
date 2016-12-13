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

class ActorNetwork(object):
    def __init__(self, sess, state_size, image_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)

        #Now create the model
        self.model , self.weights, self.state, self.image = self.create_actor_network(state_size, image_size, action_size)
        self.target_model, self.target_weights, self.target_state, self.target_image = self.create_actor_network(state_size, image_size, action_size)
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.initialize_all_variables())

    def train(self, states, images, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.image: images,
            self.action_gradient: action_grads
        })

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in xrange(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self,state_size,image_size,action_dim):
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
        I4 = Dense(HIDDEN1_UNITS, activation='relu', init=lambda shape, name: normal(shape, scale=0.01, name=name))(I3)
        print("Now we build state model")
        S = Input(shape=[state_size])
        print("Now we build concat model")
        h_0 = Dense(HIDDEN1_UNITS, activation='relu')(S)
        merged = merge([I4, h_0], mode='concat')
        h0 = Dense(HIDDEN1_UNITS, activation='relu')(merged)
        h1 = Dense(HIDDEN2_UNITS, activation='relu')(h0)
        Steering = Dense(1,activation='tanh',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)
        Acceleration = Dense(1,activation='sigmoid',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)
        Brake = Dense(1,activation='sigmoid',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)
        V = merge([Steering,Acceleration,Brake],mode='concat')
        model = Model(input=[I,S],output=V)

        return model, model.trainable_weights, S, I
