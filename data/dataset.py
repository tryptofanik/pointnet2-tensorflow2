import tensorflow as tf
from functools import partial
from sklearn.model_selection import train_test_split
from glob import glob
import sys

from .dataset_utils import parse_filename, tf_parse_filename

class ProteinDataset:
	def __init__(
		self,
		path,
		batch_size,
		n_classes,
		dataset='f199',
		shuffle_buffer=10000,
		val_split=0.15,
		n_points=1024,
		cords_channels=3,
		features_channels=0,
	):

		self.path = path
		self.batch_size = batch_size
		self.n_classes = n_classes
		self.iterator = None
		self.dataset_type = dataset
		self.shuffle_buffer = shuffle_buffer
		self.val_split = val_split
		self.n_points = n_points
		self.cords_channels = cords_channels
		self.features_channels = features_channels

		self.create_database()
		self.get_iterator()


	def create_database(self):
		# Create datasets (.map() after .batch() due to lightweight mapping fxn)

		print(f'Creating datase from {self.path}...')
		self.train_files, self.val_files, self.test_files = [], [], []

		for obj_type in glob(f'{self.path}/*/'):
			cur_files = glob(obj_type + 'train/*.npy')
			if len(cur_files) < 10:
				print(f'{obj_type} do not have enough data to split for validation and train set')
				self.train_files.extend(cur_files)
				continue
			cur_train, cur_val = train_test_split(cur_files, test_size=self.val_split, random_state=0, shuffle=True)
			self.train_files.extend(cur_train)
			self.val_files.extend(cur_val)

		for obj_type in glob(f'{self.path}/*/'):
			cur_files = glob(obj_type + 'test/*.npy')
			self.test_files.extend(cur_files)

		if len(self.train_files) == 0:
			print('No training data! Aborting.')
			raise

		print('Number of points: ', self.n_points)
		print('Number of coordinates channels:', self.cords_channels)
		print('Number of features channels:', self.features_channels)
		print('Number of classes:', self.n_classes)
		print('Number of training samples:', len(self.train_files))
		print('Number of validation samples:', len(self.val_files))
		print('Number of test samples:', len(self.test_files))

		AUTOTUNE = tf.data.experimental.AUTOTUNE

		parse_filename_filled = partial(
			parse_filename, n_points=self.n_points, cords_channels=self.cords_channels, features_channels=self.features_channels, is_test=False
		)
		tf_parse_filename_filled = partial(tf_parse_filename, parse_filename=parse_filename_filled)

		parse_filename_test_filled = partial(
			parse_filename, n_points=self.n_points, cords_channels=self.cords_channels, features_channels=self.features_channels, is_test=True
		)
		tf_parse_filename_test_filled = partial(tf_parse_filename, parse_filename=parse_filename_test_filled)

		self.train_ds = tf.data.Dataset.list_files(self.train_files)
		self.train_ds = self.train_ds.batch(self.batch_size, drop_remainder=True)
		self.train_ds = self.train_ds.map(tf_parse_filename_filled, num_parallel_calls=AUTOTUNE)
		self.train_ds = self.train_ds.prefetch(buffer_size=AUTOTUNE)

		self.val_ds = tf.data.Dataset.list_files(self.val_files)
		self.val_ds = self.val_ds.batch(self.batch_size, drop_remainder=True)
		self.val_ds = self.val_ds.map(tf_parse_filename_filled, num_parallel_calls=AUTOTUNE)

		self.test_ds = tf.data.Dataset.list_files(self.test_files)
		self.test_ds = self.test_ds.batch(self.batch_size, drop_remainder=True)
		self.test_ds = self.test_ds.map(tf_parse_filename_test_filled, num_parallel_calls=AUTOTUNE)

		print('Done!')

	def get_iterator(self):

		self.iterator = self.train_ds.__iter__()

	def reset_iterator(self):

		self.dataset.shuffle(self.shuffle_buffer)
		self.get_iterator()

	def get_batch(self):

		batch = self.iterator.next()
		return batch


class TFDataset:
	def __init__(self, path, batch_size, dataset='modelnet', shuffle_buffer=10000):

		self.path = path
		self.batch_size = batch_size
		self.iterator = None
		self.dataset_type = dataset
		self.shuffle_buffer = shuffle_buffer

		self.dataset = self.read_tfrecord(self.path, self.batch_size)

		self.get_iterator()

	def read_tfrecord(self, path, batch_size):

		dataset = tf.data.TFRecordDataset(path).shuffle(self.shuffle_buffer).batch(batch_size)
		if self.dataset_type == 'modelnet':
			dataset = dataset.map(self.extract_modelnet_fn)
		elif self.dataset_type == 'scannet':
			dataset = dataset.map(self.extract_scannet_fn)

		return dataset

	def extract_modelnet_fn(self, data_record):

		features = {'points': tf.io.VarLenFeature(tf.float32), 'label': tf.io.FixedLenFeature([], tf.int64)}

		sample = tf.io.parse_example(data_record, features)

		return sample['points'].values, sample['label']

	def extract_scannet_fn(self, data_record):

		features = {'points': tf.io.VarLenFeature(tf.float32), 'labels': tf.io.VarLenFeature(tf.int64)}

		sample = tf.io.parse_example(data_record, features)

		return sample['points'].values, sample['labels'].values

	def get_iterator(self):

		self.iterator = self.dataset.__iter__()

	def reset_iterator(self):

		self.dataset.shuffle(self.shuffle_buffer)
		self.get_iterator()

	def get_batch(self):

		while True:
			try:
				batch = self.iterator.next()
				if self.dataset_type == 'modelnet':
					pts = tf.reshape(batch[0], (self.batch_size, -1, 3))
					label = tf.reshape(batch[1], (self.batch_size, 1))
				elif self.dataset_type == 'scannet':
					pts = tf.reshape(batch[0], (self.batch_size, -1, 3))
					label = tf.reshape(batch[1], (self.batch_size, -1))
				break
			except:
				self.reset_iterator()

		return pts, label
