from __future__ import division
from __future__ import print_function
from six.moves import xrange
import chainer, argparse, os, sys, cupy, collections, six, math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from chainer import optimizers, iterators, cuda, Variable, initializers
from chainer import links as L
from chainer import functions as F
from selu import selu

class SELUModel(chainer.Chain):
	def __init__(self):
		super(SELUModel, self).__init__(
			l1=L.Linear(None, 1200),
			l2=L.Linear(None, 1200),
			l3=L.Linear(None, 10),
		)

	def __call__(self, x, apply_softmax=True):
		xp = self.xp
		out = selu(self.l1(x))
		# print(xp.mean(out.data), xp.std(out.data))
		out = selu(self.l2(out))
		# print(xp.mean(out.data), xp.std(out.data))
		out = self.l3(out)
		# print(out)
		# print(xp.mean(out.data), xp.std(out.data))
		if apply_softmax:
			out = F.softmax(out)
		return out

class ReLUBatchNormModel(chainer.Chain):
	def __init__(self):
		super(ReLUBatchNormModel, self).__init__(
			l1=L.Linear(None, 1200),
			l2=L.Linear(None, 1200),
			l3=L.Linear(None, 10),
			bn1=L.BatchNormalization(1200),
			bn2=L.BatchNormalization(1200),
		)

	def __call__(self, x, apply_softmax=True):
		out = self.bn1(F.relu(self.l1(x)))
		out = self.bn2(F.relu(self.l2(out)))
		out = self.l3(out)
		if apply_softmax:
			out = F.softmax(out)
		return out

class DeepModel(chainer.Chain):
	def __init__(self, num_layers=8):
		super(DeepModel, self).__init__(
			logits=L.Linear(None, 10),
		)
		self.num_layers = num_layers
		self.activations = []
		for idx in xrange(num_layers):
			self.add_link("layer_%s" % idx, L.Linear(None, 1000))

class SELUDeepModel(DeepModel):
	name = "SELU"
	def __call__(self, x, apply_softmax=True):
		del self.activations[:]
		xp = self.xp
		out = x
		for idx in xrange(self.num_layers):
			layer = getattr(self, "layer_%s" % idx)
			out = selu(layer(out))
			if chainer.config.train:
				self.activations.append(out)
		out = self.logits(out)
		if apply_softmax:
			out = F.softmax(out)
		return out

class ReLUBatchnormDeepModel(DeepModel):
	name = "ReLU+BatchNorm"
	def __init__(self, num_layers=8):
		super(ReLUBatchnormDeepModel, self).__init__(num_layers=num_layers)
		for idx in xrange(num_layers):
			self.add_link("bn_%s" % idx, L.BatchNormalization(1000))

	def __call__(self, x, apply_softmax=True):
		del self.activations[:]
		xp = self.xp
		out = x
		for idx in xrange(self.num_layers):
			layer = getattr(self, "layer_%s" % idx)
			batchnorm = getattr(self, "bn_%s" % idx)
			out = batchnorm(F.relu(layer(out)))
			if chainer.config.train:
				self.activations.append(out)
		out = self.logits(out)
		if apply_softmax:
			out = F.softmax(out)
		return out

class RELUDeepModel(DeepModel):
	name = "ReLU"
	def __call__(self, x, apply_softmax=True):
		del self.activations[:]
		xp = self.xp
		out = x
		for idx in xrange(self.num_layers):
			layer = getattr(self, "layer_%s" % idx)
			out = F.relu(layer(out))
			if chainer.config.train:
				self.activations.append(out)
		out = self.logits(out)
		if apply_softmax:
			out = F.softmax(out)
		return out

class ELUDeepModel(DeepModel):
	name = "ELU"
	def __call__(self, x, apply_softmax=True):
		del self.activations[:]
		xp = self.xp
		out = x
		for idx in xrange(self.num_layers):
			layer = getattr(self, "layer_%s" % idx)
			out = F.elu(layer(out))
			if chainer.config.train:
				self.activations.append(out)
		out = self.logits(out)
		if apply_softmax:
			out = F.softmax(out)
		return out

def compute_entropy(p):
	if p.ndim == 2:
		return -F.sum(p * F.log(p + 1e-16), axis=1)
	return -F.sum(p * F.log(p + 1e-16))

def compute_marginal_entropy(p):
	return compute_entropy(F.mean(p, axis=0))

def compute_kld(p, q):
	assert len(p) == len(q)
	return F.reshape(F.sum(p * (F.log(p + 1e-16) - F.log(q + 1e-16)), axis=1), (-1, 1))

def get_unit_vector(v):
	xp = cuda.get_array_module(v)
	if v.ndim == 4:
		return v / (xp.sqrt(xp.sum(v ** 2, axis=(1,2,3))).reshape((-1, 1, 1, 1)) + 1e-16)
	return v / (xp.sqrt(xp.sum(v ** 2, axis=1)).reshape((-1, 1)) + 1e-16)

def compute_lds(model, x, xi=10, eps=1, Ip=1):
	xp = cuda.get_array_module(x)
	y1 = model(x, apply_softmax=True)
	y1.unchain_backward()
	d = Variable(get_unit_vector(np.random.normal(size=x.shape).astype(np.float32)))
	if xp is cupy:
		d.to_gpu()

	for i in xrange(Ip):
		y2 = model(x + xi * d, apply_softmax=True)
		kld = F.sum(compute_kld(y1, y2))
		kld.backward()
		d = Variable(get_unit_vector(d.grad))
	
	y2 = model(x + eps * d, apply_softmax=True)
	return -compute_kld(y1, y2)

def get_mnist():
	mnist_train, mnist_test = chainer.datasets.get_mnist()
	train_data, train_label = [], []
	test_data, test_label = [], []
	for data in mnist_train:
		train_data.append(data[0])
		train_label.append(data[1])
	for data in mnist_test:
		test_data.append(data[0])
		test_label.append(data[1])
	train_data = np.asanyarray(train_data, dtype=np.float32)
	test_data = np.asanyarray(test_data, dtype=np.float32)
	train_data = (train_data - np.mean(train_data)) / np.std(train_data)
	test_data = (test_data - np.mean(test_data)) / np.std(test_data)
	return (train_data, np.asanyarray(train_label, dtype=np.int32)), (test_data, np.asanyarray(test_label, dtype=np.int32))

def compute_clustering_accuracy(model, x, t, num_clusters=10):
	xp = model.xp
	batches = xp.split(x, len(x) // 100)
	probs = None
	for batch in batches:
		p = F.softmax(model(batch, apply_softmax=True)).data
		probs = p if probs is None else xp.concatenate((probs, p), axis=0)
	labels_predict = xp.argmax(probs, axis=1)
	predict_counts = np.zeros((10, num_clusters), dtype=int)

	for i in xrange(len(x)):
		label_predict = int(labels_predict[i])
		label_true = int(t[i])
		predict_counts[label_true, label_predict] += 1

	indices = np.argmax(predict_counts, axis=0)
	match_count = np.zeros((10,), dtype=np.float32)
	for i in xrange(num_clusters):
		assinged_label = indices[i]
		match_count[assinged_label] += predict_counts[assinged_label][i]

	accuracy = np.sum(match_count) / len(x)
	return predict_counts, accuracy

def compute_classification_accuracy(model, x, t):
	xp = model.xp
	batches = xp.split(x, len(x) // 100)
	scores = None
	for batch in batches:
		p = F.softmax(model(batch, apply_softmax=False)).data
		scores = p if scores is None else xp.concatenate((scores, p), axis=0)
	return float(F.accuracy(scores, Variable(t)).data)

def plot_activations(model, x, out_dir):
	try:
		os.mkdir(out_dir)
	except:
		pass

	if isinstance(model, DeepModel):
		sns.set(font_scale=0.5)
		fig = plt.figure()
		num_layers = model.num_layers
		
		with chainer.using_config("Train", True):
			xp = model.xp
			batches = xp.split(x, len(x) // 200)
			num_layers = model.num_layers
			layer_activations = [None] * num_layers
			for batch_idx, batch in enumerate(batches):
				sys.stdout.write("\rplotting {}/{}".format(batch_idx + 1, len(batches)))
				sys.stdout.flush()
				logits = model(batch)
				for layer_idx, activations in enumerate(model.activations):
					data = cuda.to_cpu(activations.data).reshape((-1,))
					# append
					pool = layer_activations[layer_idx]
					pool = data if pool is None else np.concatenate((pool, data))
					layer_activations[layer_idx] = pool

			fig, axes = plt.subplots(1, num_layers)
			for layer_idx, (activations, ax) in enumerate(zip(layer_activations, axes)):
				ax.hist(activations, bins=20)
				ax.set_xlim([-5, 5])
				ax.set_ylim([0, 60000 * 100])
				ax.get_yaxis().set_major_formatter(mtick.FormatStrFormatter("%.e"))

			fig.suptitle("%s Activation Distribution" % model.__class__.name)
			plt.savefig(os.path.join(out_dir, "activation.png"), dpi=350)

def train_supervised(args):
	mnist_train, mnist_test = get_mnist()

	# seed
	np.random.seed(args.seed)
	if args.gpu_device != -1:
		cuda.cupy.random.seed(args.seed)
		
	# init model
	model = None
	if args.model.lower() == "selu":
		model = SELUDeepModel()
	elif args.model.lower() == "relu":
		model = RELUDeepModel()
	elif args.model.lower() == "bn":
		model = ReLUBatchnormDeepModel()
	elif args.model.lower() == "elu":
		model = ELUDeepModel()
	assert model is not None
	if args.gpu_device >= 0:
		cuda.get_device(args.gpu_device).use()
		model.to_gpu()
	xp = model.xp

	# init optimizer
	optimizer = optimizers.Adam(alpha=args.learning_rate, beta1=0.9)
	optimizer.setup(model)

	train_data, train_label = mnist_train
	test_data, test_label = mnist_test
	if args.gpu_device >= 0:
		train_data = cuda.to_gpu(train_data)
		train_label = cuda.to_gpu(train_label)
		test_data = cuda.to_gpu(test_data)
		test_label = cuda.to_gpu(test_label)
	train_loop = len(train_data) // args.batchsize
	train_indices = np.arange(len(train_data))

	# training cycle
	for epoch in xrange(1, args.epoch):
		np.random.shuffle(train_indices)	# shuffle data
		sum_loss = 0

		with chainer.using_config("Train", True):
			# loop over all batches
			for itr in xrange(train_loop):
				# sample minibatch
				batch_range = np.arange(itr * args.batchsize, min((itr + 1) * args.batchsize, len(train_data)))
				x = train_data[train_indices[batch_range]]
				t = train_label[train_indices[batch_range]]

				# to gpu
				if model.xp is cuda.cupy:
					x = cuda.to_gpu(x)
					t = cuda.to_gpu(t)

				logits = model(x, apply_softmax=False)
				loss = F.softmax_cross_entropy(logits, Variable(t))

				# update weights
				optimizer.update(lossfun=lambda: loss)

				if itr % 50 == 0:
					sys.stdout.write("\riteration {}/{}".format(itr, train_loop))
					sys.stdout.flush()
				sum_loss += float(loss.data)

		with chainer.using_config("Train", False):
			accuracy_train = compute_classification_accuracy(model, train_data, train_label)
			accuracy_test = compute_classification_accuracy(model, test_data, test_label)

		sys.stdout.write("\r\033[2KEpoch {} - loss: {:.5f} - acc: {:.4f} (train), {:.4f} (test)\n".format(epoch, sum_loss / train_loop, accuracy_train, accuracy_test))
		sys.stdout.flush()

	# plot activations
	plot_activations(model, test_data, args.model)

def train_unsupervised(args):
	try:
		os.mkdir(args.model)
	except:
		pass
	mnist_train, mnist_test = get_mnist()

	# seed
	np.random.seed(args.seed)
	if args.gpu_device != -1:
		cuda.cupy.random.seed(args.seed)
		
	# init model
	model = SELUDeepModel()
	if args.gpu_device >= 0:
		cuda.get_device(args.gpu_device).use()
		model.to_gpu()
	xp = model.xp

	# init optimizer
	optimizer = optimizers.Adam(alpha=args.learning_rate, beta1=0.9)
	optimizer.setup(model)

	# IMSAT hyperparameters
	lam = 0.2
	mu = 4.0

	train_data, train_label = mnist_train
	test_data, test_label = mnist_test
	if args.gpu_device >= 0:
		train_data = cuda.to_gpu(train_data)
		train_label = cuda.to_gpu(train_label)
		test_data = cuda.to_gpu(test_data)
		test_label = cuda.to_gpu(test_label)
	train_loop = len(train_data) // args.batchsize
	train_indices = np.arange(len(train_data))

	# training cycle
	for epoch in xrange(1, args.epoch):
		np.random.shuffle(train_indices)	# shuffle data
		sum_loss = 0
		sum_hy = 0
		sum_hy_x = 0
		sum_Rsat = 0

		with chainer.using_config("Train", True):
			# loop over all batches
			for itr in xrange(train_loop):
				# sample minibatch
				batch_range = np.arange(itr * args.batchsize, (itr + 1) * args.batchsize)
				x = train_data[train_indices[batch_range]]
				t = train_label[train_indices[batch_range]]

				# to gpu
				if model.xp is cuda.cupy:
					x = cuda.to_gpu(x)
					t = cuda.to_gpu(t)

				p = model(x, apply_softmax=True)

				# compute loss
				hy = compute_marginal_entropy(p)
				hy_x = F.mean(compute_entropy(p))
				Rsat = -F.mean(compute_lds(model, x))
				loss = Rsat - lam * (mu * hy - hy_x)
				loss = F.softmax_cross_entropy(model(x, apply_softmax=False), Variable(t))

				# update weights
				optimizer.update(lossfun=lambda: loss)

				if itr % 50 == 0:
					sys.stdout.write("\riteration {}/{}".format(itr, train_loop))
					sys.stdout.flush()
				sum_loss += float(loss.data)
				sum_hy += float(hy.data)
				sum_hy_x += float(hy_x.data)
				sum_Rsat += float(Rsat.data)

		with chainer.using_config("Train", False):
			counts_train, accuracy_train = compute_clustering_accuracy(model, train_data, train_label)
			test_data, test_label = mnist_test
			counts_test, accuracy_test = compute_clustering_accuracy(model, test_data, test_label)

		sys.stdout.write("\r\033[2KEpoch {} - loss: {:.5f} - acc: {:.4f} (train), {:.4f} (test) - hy: {:.4f} - hy_x: {:.4f} - Rsat: {:.4f}\n".format(epoch, sum_loss / train_loop, accuracy_train, accuracy_test, sum_hy / train_loop, sum_hy_x / train_loop, sum_Rsat / train_loop))
		sys.stdout.flush()
		print(counts_train)
		print(counts_test)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", "-m", type=str, default="selu")
	parser.add_argument("--gpu-device", "-g", type=int, default=0)
	parser.add_argument("--learning-rate", "-lr", type=float, default=0.002)
	parser.add_argument("--epoch", "-e", type=int, default=50)
	parser.add_argument("--batchsize", "-b", type=int, default=256)
	parser.add_argument("--seed", "-seed", type=int, default=0)
	args = parser.parse_args()
	train_supervised(args)
	# train_unsupervised(args)