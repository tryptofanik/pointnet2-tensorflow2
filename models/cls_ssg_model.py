import os
import sys

sys.path.insert(0, './')

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Lambda

from pnet2_layers.layers import Pointnet_SA, Pointnet_SA_MSG


class CLS_SSG_Model(Model):

	def __init__(self, batch_size, num_points, num_classes, cords_channels=3, features_channels=0, bn=False, activation=tf.nn.relu):
		super(CLS_SSG_Model, self).__init__()

		self.activation = activation
		self.batch_size = batch_size
		self.num_points = num_points
		self.num_classes = num_classes
		self.cords_channels = cords_channels
		self.features_channels = features_channels
		self.bn = bn
		self.keep_prob = 0.5

		self.kernel_initializer = 'glorot_normal'
		self.kernel_regularizer = None

		self.init_network()


	def init_network(self):

		self.layer1 = Pointnet_SA(
			npoint=512,
			cords_channels=self.cords_channels,
			features_channels=self.features_channels,
			radius=0.2,
			nsample=32,
			mlp=[64, 64, 128],
			group_all=False,
			activation=self.activation,
			bn=self.bn
		)

		self.layer2 = Pointnet_SA(
			npoint=128,
			cords_channels=self.cords_channels,
			features_channels=self.features_channels,
			radius=0.4,
			nsample=64,
			mlp=[128, 128, 256],
			group_all=False,
			activation=self.activation,
			bn=self.bn
		)

		self.layer3 = Pointnet_SA(
			npoint=None,
			cords_channels=self.cords_channels,
			features_channels=self.features_channels,
			radius=None,
			nsample=None,
			mlp=[256, 512, 1024],
			group_all=True,
			activation=self.activation,
			bn=self.bn
		)

		self.dense1 = Dense(512, activation=self.activation)
		if self.bn: self.bn_fc1 = BatchNormalization()

		self.dropout1 = Dropout(self.keep_prob)

		self.dense2 = Dense(128, activation=self.activation)
		if self.bn: self.bn_fc2 = BatchNormalization()

		self.dropout2 = Dropout(self.keep_prob)

		self.dense3 = Dense(self.num_classes, activation=tf.nn.softmax)



	def call(self, input, training=True):

		xyz = Lambda(lambda x: input[:, :, :self.cords_channels])(input)
		if self.features_channels:
			points = Lambda(lambda x: input[:, :, self.cords_channels:])(input)
		else:
			points = None

		xyz, points = self.layer1(xyz, points, training=training)
		xyz, points = self.layer2(xyz, points, training=training)
		xyz, points = self.layer3(xyz, points, training=training)

		net = tf.reshape(points, (self.batch_size, -1))

		net = self.dense1(net)
		if self.bn: net = self.bn_fc1(net, training=training)
		net = self.dropout1(net)

		net = self.dense2(net)
		if self.bn: net = self.bn_fc2(net, training=training)
		net = self.dropout2(net)

		pred = self.dense3(net)

		return pred
