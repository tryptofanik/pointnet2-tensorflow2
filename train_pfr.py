import os
import sys
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

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


def update_metrics(metrics, pred, y_truth, loss):
	metrics['acc'].update_state(y_truth, pred)
	metrics['loss'].update_state([loss])
	metrics['precision'].update_state(y_truth, pred)
	metrics['recall'].update_state(y_truth, pred)
	metrics['acc'].update_state(y_truth, pred)
	metrics['false_neg'].update_state(y_truth, pred)
	metrics['false_pos'].update_state(y_truth, pred)
	metrics['true_neg'].update_state(y_truth, pred)
	metrics['true_pos'].update_state(y_truth, pred)


def train_step(optimizer, model, loss_object, metrics, x_train, y_train):

	with tf.GradientTape() as tape:

		pred = model(x_train)
		loss = loss_object(y_train, pred)

	update_metrics(metrics, pred, y_train, loss)

	gradients = tape.gradient(loss, model.trainable_variables)
	optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test_step(optimizer, model, loss_object, metrics, x_test, y_test):

	with tf.GradientTape():

		pred = model(x_test)
		loss = loss_object(y_test, pred)

	update_metrics(metrics, pred, y_test, loss)


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

	metrics = {}
	for ds in ['train', 'val', 'test']:
		metrics[ds] = {}
		metrics[ds]['acc'] = tf.keras.metrics.CategoricalAccuracy()
		metrics[ds]['loss'] = tf.keras.metrics.Mean()
		metrics[ds]['precision'] = tf.keras.metrics.Precision()
		metrics[ds]['recall'] = tf.keras.metrics.Recall()
		metrics[ds]['acc'] = tf.keras.metrics.CategoricalAccuracy()
		metrics[ds]['false_neg'] = tf.keras.metrics.FalseNegatives()
		metrics[ds]['false_pos'] = tf.keras.metrics.FalsePositives()
		metrics[ds]['true_neg'] = tf.keras.metrics.TrueNegatives()
		metrics[ds]['true_pos'] = tf.keras.metrics.TruePositives()

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

			train_step(optimizer, model, loss_object, metrics['train'], x_train, y_train)

			lr.assign(get_lr(**LR_ARGS, step=step))

			step += 1

		if config['wandb']:
			wandb.log(
				{**{'train_' + k: v.result().numpy() for k, v in metrics['train'].items()}, 'step': step, 'lr': lr.numpy()}
			)

		for metric in metrics['train'].values():
			metric.reset_states()

		if epoch % config['val_freq'] == 0:

			for x_val, y_val in val_ds:

				test_step(optimizer, model, loss_object, metrics['val'], x_val, y_val)

			if config['wandb']:
				wandb.log({**{'val_' + k: v.result().numpy() for k, v in metrics['val'].items()}, 'epoch': epoch, 'lr': lr.numpy()})

			if metrics['val']['acc'].result().numpy() > best_acc:
				best_acc = metrics['val']['acc'].result().numpy()
				model.save_weights('model/checkpoints/' + INIT_TIMESTAMP + '/epoch-' + str(epoch), save_format='tf')

			for metric in metrics['val'].values():
				metric.reset_states()

	# Evaluate
	size = 0

	confusion_mat = np.zeros((params['num_classes'], params['num_classes']), dtype=np.float32)
	for x_test, y_test in test_ds:
		size += x_test.shape[0]

		pred = model(x_test, training=False)
		loss = loss_object(y_train, pred)

		update_metrics(metrics['test'], pred, y_test, loss)

		max_idxs = tf.math.argmax(pred, axis=1)
		for true_class, pred_class in zip(np.argwhere(y_test)[:, 1], max_idxs.numpy()):
			confusion_mat[true_class, pred_class] += 1

	row_norm = np.sum(confusion_mat, axis=1)
	row_norm = np.expand_dims(row_norm, axis=1)
	row_norm = np.repeat(row_norm, params['num_classes'], axis=1)
	confusion_mat /= row_norm

	mask = confusion_mat < 0.05
	plt.figure(figsize=(11, 10.5))  # width by height
	ax = sns.heatmap(confusion_mat, annot=True, annot_kws={'size': 9},
						fmt='.1f', cbar=False, cmap='binary', mask=mask, linecolor='black', linewidths=0.5)
	ax.xaxis.set_ticks_position('top')
	ax.xaxis.set_label_position('top')
	ax.set_ylabel('True Class')
	ax.set_xlabel('Predicted Class')
	ax.spines['top'].set_visible(True)
	plt.yticks(rotation=0)
	plt.savefig('figs/confusion_matrix.png', bbox_inches='tight')

	if config['wabd']:
		wandb.log({'test_' + k: v.result().numpy() for k, v in metrics['test'].items()})


if __name__ == '__main__':

	config = {
		'dataset_dir': '/scidatalg/ar/scaled_splited7',
		'log_dir': 'logs',
		'log_code': 'ssg_1',
		# 'log_freq': 10,
		'val_freq': 1,
		'test_freq': 100,
		'wandb': True,
	}

	params = {
		'batch_size': 35,
		'num_points': 2048,
		'num_classes': 198,
		'cords_channels': 3,
		'features_channels': 4,
		'val_split': 0.1,
		'lr': 0.01,
		'lr_decay_steps': 3000,
		'lr_decay_rate': 0.7,
		'epochs': 2000,
		'msg': False,
		'bn': False,
	}

	train(config, params)
