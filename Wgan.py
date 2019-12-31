# example of a wgan for generating handwritten digits
from numpy import expand_dims
from numpy import mean
from numpy import ones, array
from numpy import hstack, vstack
from numpy.random import randn
from numpy.random import randint
from keras.datasets.mnist import load_data
from keras import backend
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.initializers import RandomNormal
from keras.constraints import Constraint
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects
from matplotlib import pyplot
import sys
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pylab import savefig

ALPHA = 0.001

# clip model weights to a given hypercube
class ClipConstraint(Constraint):
	# set clip value when initialized
	def __init__(self, clip_value):
		self.clip_value = clip_value

	# clip model weights to hypercube
	def __call__(self, weights):
		return backend.clip(weights, -self.clip_value, self.clip_value)

	# get the config
	def get_config(self):
		return {'clip_value': self.clip_value}

# calculate wasserstein loss
def wasserstein_loss(y_true, y_pred):
	return backend.mean(y_true * y_pred)

# threshold function used for generating samples for behavioral learning data
def custom_activation(x):
	return np.where(x > 0, 1 , 0)

get_custom_objects().update({'custom_activation': Activation(custom_activation)})

# define the standalone critic model
def define_critic(in_shape):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# weight constraint
	const = ClipConstraint(0.01)
	# define model
	model = Sequential()

	model.add(Dense(16, kernel_initializer=init, kernel_constraint=const, input_shape=(in_shape,)))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))

	model.add(Dense(16, kernel_initializer=init, kernel_constraint=const))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))
	# scoring, linear activation
	model.add(Dense(1))
	# compile model
	opt = RMSprop(lr=ALPHA)
	model.compile(loss=wasserstein_loss, optimizer=opt)
	return model

# define the standalone generator model
def define_generator(output_dim, latent_dim):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# define model
	model = Sequential()

	model.add(Dense(32, kernel_initializer=init, input_shape=(latent_dim,)))
	model.add(BatchNormalization()) # not in provided code, but I added this
	model.add(LeakyReLU(alpha=0.2))

	model.add(Dense(16, kernel_initializer=init))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))

	model.add(Dense(16, kernel_initializer=init))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))

	# output layer
	model.add(Dense(output_dim, kernel_initializer=init))

	# for behavioral data, map outputs to either 1 or 0
	if dataset_name == 'Behavioral':
		model.add(Activation(custom_activation, name='SpecialActivation'))

	return model

# define the combined generator and critic model, for updating the generator
def define_gan(generator, critic):
	# make weights in the critic not trainable
	critic.trainable = False
	# connect them
	model = Sequential()
	# add generator
	model.add(generator)
	# add the critic
	model.add(critic)
	# compile model
	opt = RMSprop(lr=ALPHA)
	model.compile(loss=wasserstein_loss, optimizer=opt)
	critic.trainable = True
	return model

# load images
def load_real_samples(which_dataset):
	# load dataset
	if which_dataset == 0:
		df = pd.read_csv('./data/synDist.csv')
	elif which_dataset == 1:
		df = pd.read_csv('./data/behavioral.csv')
	else: # which_dataset == 2
		df = pd.read_csv('./data/pops_and_recorded.csv')
	return df.values

# select real samples
def generate_real_samples(dataset, n_samples):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# select samples
	X = dataset[ix]
	# generate class labels, -1 for 'real'
	y = -ones((n_samples, 1))
	return X, y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	X = generator.predict(x_input)
	# create class labels with 1.0 for 'fake'
	y = ones((n_samples, 1))
	return X, y

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, latent_dim, real_samples):
	# prepare fake examples
	n_samples = real_samples.shape[0]
	X, _ = generate_fake_samples(g_model, latent_dim, n_samples)

	labels = ['Real'] * real_samples.shape[0] + ['Generated'] * X.shape[0]
	real_and_fake = vstack((real_samples, X))
	#rf_two_dim = TSNE(n_components=2).fit_transform(real_and_fake)
	rf_two_dim = pca.transform(real_and_fake)

	print(rf_two_dim.shape)
	labels = array([[num] for num in labels])
	print(labels.shape)
	df = pd.DataFrame(data=rf_two_dim,
					  columns=['X1', 'X2'])
	df['Data Type'] = labels
	print('DATAFRAME MADE')
	print(df)

	ax = sns.scatterplot(x="X1", y="X2", hue="Data Type", data=df)
	print('SCATTERPLOT MADE')

	# save plot to file
	filename1 = './output/generated_plot_%04d_%s.png' % (step+1, dataset_name)
	figure = ax.get_figure()
	print('FIGURE RETRIEVED')
	figure.savefig(filename1, dpi=400)
	print('SCATTERPLOT SAVED')
	pyplot.close()
	# save the generator model
	filename2 = './output/model_%04d_%s.h5' % (step+1, dataset_name)
	g_model.save(filename2)
	print('MODEL SAVED')
	print('>Saved: %s and %s' % (filename1, filename2))

# create a line plot of loss for the gan and save to file
def plot_history(d1_hist, d2_hist, g_hist):
	# plot history
	pyplot.plot(d1_hist, label='crit_real')
	pyplot.plot(d2_hist, label='crit_fake')
	pyplot.plot(g_hist, label='gen')
	pyplot.legend()
	pyplot.savefig('./output/plot_line_plot_loss_{0}.png'.format(dataset_name))
	pyplot.close()

# train the generator and critic
def train(g_model, c_model, gan_model, dataset, latent_dim, n_epochs=10, n_batch=64, n_critic=5):
	# calculate the number of batches per training epoch
	bat_per_epo = int(dataset.shape[0] / n_batch)
	# calculate the number of training iterations
	#n_steps = bat_per_epo * n_epochs
	n_steps = 20000
	# calculate the size of half a batch of samples
	half_batch = int(n_batch / 2)
	# lists for keeping track of loss
	c1_hist, c2_hist, g_hist = list(), list(), list()
	# manually enumerate epochs
	for i in range(n_steps):
		# update the critic more than the generator
		c1_tmp, c2_tmp = list(), list()
		for _ in range(n_critic):
			# get randomly selected 'real' samples
			X_real, y_real = generate_real_samples(dataset, half_batch)
			# update critic model weights
			c_loss1 = c_model.train_on_batch(X_real, y_real)
			c1_tmp.append(c_loss1)
			# generate 'fake' examples
			X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
			# update critic model weights
			c_loss2 = c_model.train_on_batch(X_fake, y_fake)
			c2_tmp.append(c_loss2)
		# store critic loss
		c1_hist.append(mean(c1_tmp))
		c2_hist.append(mean(c2_tmp))
		# prepare points in latent space as input for the generator
		X_gan = generate_latent_points(latent_dim, n_batch)
		# create inverted labels for the fake samples
		y_gan = -ones((n_batch, 1))
		# update the generator via the critic's error
		g_loss = gan_model.train_on_batch(X_gan, y_gan)
		g_hist.append(g_loss)
		# summarize loss on this batch
		print('>%d, c1=%.3f, c2=%.3f g=%.3f' % (i+1, c1_hist[-1], c2_hist[-1], g_loss))
		# evaluate the model performance every 'epoch'
		#if (i+1) % bat_per_epo == 0:
		if (i+1) % (n_steps // 10) == 0:
			summarize_performance(i, g_model, latent_dim, X_real)
	# line plots of loss
	plot_history(c1_hist, c2_hist, g_hist)


# load correct dataset
# 0 for synthetic
# 1 for behavioral learning experiment
# 2 for stop and frisk
which_dataset = int(sys.argv[1])
print('Which dataset: '),
if which_dataset == 0:
	dataset_name = 'Synthetic'
elif which_dataset == 1:
	dataset_name = 'Behavioral'
else:
	dataset_name = 'StopAndFrisk'
print(dataset_name)
dataset = load_real_samples(which_dataset)
print(dataset.shape)

# size of the latent space
latent_dim = 50
num_features = dataset.shape[1]
# create the critic
critic = define_critic(num_features)
# create the generator
generator = define_generator(num_features, latent_dim)
# create the gan
gan_model = define_gan(generator, critic)

pca = PCA(n_components=2).fit(dataset)

# train model
train(generator, critic, gan_model, dataset, latent_dim)
