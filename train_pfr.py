import os
import sys
import datetime

sys.path.insert(0, '.data/')
sys.path.insert(0, './')

import numpy as np
import tensorflow as tf

from data.dataset import ProteinDataset
from models.cls_msg_model import CLS_MSG_Model
from models.cls_ssg_model import CLS_SSG_Model
from models.cls_basic_model import Pointnet_Model

tf.random.set_seed(42)


def get_timestamp():
	timestamp = str(datetime.datetime.now())[:16]
	timestamp = timestamp.replace('-', '')
	timestamp = timestamp.replace(' ', '_')
	timestamp = timestamp.replace(':', '')
	return timestamp


INIT_TIMESTAMP = get_timestamp()


def train_step(optimizer, model, loss_object, train_loss, train_acc, x_train, y_train):

	with tf.GradientTape() as tape:

		pred = model(x_train)
		loss = loss_object(y_train, pred)

	train_loss.update_state([loss])
	train_acc.update_state(y_train, pred)

	gradients = tape.gradient(loss, model.trainable_variables)
	optimizer.apply_gradients(zip(gradients, model.trainable_variables))

	return train_loss, train_acc


def test_step(optimizer, model, loss_object, test_loss, test_acc, test_pts, test_labels):

	with tf.GradientTape() as tape:

		pred = model(test_pts)
		loss = loss_object(test_labels, pred)

	test_loss.update_state([loss])
	test_acc.update_state(test_labels, pred)

	return test_loss, test_acc


def get_lr(initial_learning_rate, decay_steps, decay_rate, step, staircase=False, warm_up=True):
	if warm_up:
		coeff1 = min(1.0, step / 2000)
	else:
		coeff1 = 1.0

	if staircase:
		coeff2 = decay_rate ** (step // decay_steps)
	else:
		coeff2 = decay_rate ** (step / decay_steps)

	current = initial_learning_rate * coeff1 * coeff2
	return current


def train(config, params):

	if config['wandb']:
		if os.environ['WANDB_API_KEY']:
			import wandb

			wandb.init(project='pointnet_pfr', name=INIT_TIMESTAMP)

	if params['msg']:
		model = CLS_MSG_Model(
			batch_size=params['batch_size'],
			num_points=params['num_points'],
			num_classes=params['num_classes'],
			bn=params['bn'])
	else:
		model = CLS_SSG_Model(
			batch_size=params['batch_size'],
			num_points=params['num_points'],
			num_classes=params['num_classes'],
			cords_channels=params['cords_channels'],
			features_channels=params['features_channels'],
			bn=params['bn'])

	model.build(
		input_shape=(params['batch_size'], params['num_points'], params['cords_channels'] + params['features_channels'])
	)
	print(model.summary())
	print('[info] model training...')

	LR_ARGS = {
		'initial_learning_rate': params['lr'],
		'decay_steps': params['lr_decay_steps'],
		'decay_rate': params['lr_decay_rate'],
		'staircase': False,
		'warm_up': True,
	}

	lr = tf.Variable(get_lr(**LR_ARGS, step=0), trainable=False)
	optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
	loss_object = tf.keras.losses.CategoricalCrossentropy()

	train_loss = tf.keras.metrics.Mean()
	val_loss = tf.keras.metrics.Mean()
	test_loss = tf.keras.metrics.Mean()

	train_acc = tf.keras.metrics.SparseCategoricalAccuracy()
	val_acc = tf.keras.metrics.SparseCategoricalAccuracy()
	test_acc = tf.keras.metrics.SparseCategoricalAccuracy()

	train_prec = tf.keras.metrics.Precision()
	val_prec = tf.keras.metrics.Precision()
	test_prec = tf.keras.metrics.Precision()

	train_recall = tf.keras.metrics.Recall()
	val_recall = tf.keras.metrics.Recall()
	test_recall = tf.keras.metrics.Recall()

	database_manager = ProteinDataset(
		path=config['dataset_dir'],
		batch_size=params['batch_size'],
		n_classes=params['num_classes'],
		dataset='f199',
		val_split=params['val_split'],
		n_points=params['num_points'],
		cords_channels=params['cords_channels'],
		features_channels=params['features_channels'],
	)

	train_ds = database_manager.train_ds
	val_ds = database_manager.val_ds
	test_ds = database_manager.test_ds

	# train_summary_writer = tf.summary.create_file_writer(os.path.join(config['log_dir'], config['log_code'], 'train'))

	# test_summary_writer = tf.summary.create_file_writer(os.path.join(config['log_dir'], config['log_code'], 'test'))

	step = 0
	best_acc = 0

	for epoch in range(params['epochs']):

		for x_train, y_train in train_ds:

			train_loss, train_acc = train_step(optimizer, model, loss_object, train_loss, train_acc, x_train, y_train)

			if config['wandb']:
				wandb.log(
					{'tain_loss': train_loss, 'train_acc': train_acc, 'step': step, 'lr': lr.numpy(),}
				)

			step += 1

		if epoch % config['val_freq'] == 0:
			for x_val, y_val in val_ds:

				val_loss, val_acc = test_step(optimizer, model, loss_object, val_loss, val_acc, x_val, y_val)

				if config['wabd']:
					wandb.log({'val_loss': val_loss, 'val_acc': val_acc, 'epoch': epoch, 'lr': lr.numpy()})

			if val_acc > best_acc:
				best_acc = val_acc
				model.save_weights('model/checkpoints/' + INIT_TIMESTAMP + '/epoch-' + str(epoch), save_format='tf')

	# while True:

	# 	x_train, y_train = train_ds.get_batch()

	# 	loss, train_acc = train_step(optimizer, model, loss_object, train_loss, train_acc, x_train, y_train)

	# 	with train_summary_writer.as_default():

	# 		if optimizer.iterations % config['log_freq'] == 0:
	# 			tf.summary.scalar('loss', train_loss.result(), step=optimizer.iterations)
	# 			tf.summary.scalar('accuracy', train_acc.result(), step=optimizer.iterations)

	# 	if optimizer.iterations % config['test_freq'] == 0:

	# 		test_pts, test_labels = val_ds.get_batch()

	# 		test_loss, test_acc = test_step(optimizer, model, loss_object, test_loss, test_acc, test_pts, test_labels)

	# 		with test_summary_writer.as_default():

	# 			tf.summary.scalar('loss', test_loss.result(), step=optimizer.iterations)
	# 			tf.summary.scalar('accuracy', test_acc.result(), step=optimizer.iterations)


if __name__ == '__main__':

	config = {
		# 'dataset_dir': '/scidatalg/ar/scaled_splited7',
		'dataset_dir': '/content/scaled_splited',
		'log_dir': 'logs',
		'log_code': 'ssg_1',
		# 'log_freq': 10,
		'val_freq': 1,
		'test_freq': 100,
		'wandb': True,
	}

	params = {
		'batch_size': 32,
		'num_points': 2048,
		'num_classes': 198,
		'cords_channels': 3,
		'features_channels': 0,
		'val_split': 0.1,
		'lr': 0.01,
		'lr_decay_steps': 7000,
		'lr_decay_rate': 0.7,
		'epochs': 100,
		'msg': False,
		'bn': False,
	}

	train(config, params)

