from __future__ import division
from __future__ import print_function
from six.moves import xrange
import chainer, argparse, os, sys, cupy, collections, six, math
import numpy as np
from chainer import optimizers, iterators, cuda, Variable, initializers
from chainer import links as L
from chainer import functions as F
from selu import selu

def _sum_sqnorm(arr):
	sq_sum = collections.defaultdict(float)
	for x in arr:
		with cuda.get_device_from_array(x) as dev:
			x = x.ravel()
			s = x.dot(x)
			sq_sum[int(dev)] += s
	return sum([float(i) for i in six.itervalues(sq_sum)])
	
class GradientClipping(object):
	name = 'GradientClipping'

	def __init__(self, threshold):
		self.threshold = threshold

	def __call__(self, opt):
		norm = np.sqrt(_sum_sqnorm([p.grad for p in opt.target.params(False)]))
		assert norm != 0
		rate = self.threshold / norm
		if rate < 1:
			for param in opt.target.params(False):
				grad = param.grad
				with cuda.get_device_from_array(grad):
					grad *= rate

class SeluModel(chainer.Chain):
	def __init__(self):
		super(SeluModel, self).__init__(
			l1=L.Linear(784, 1200),
			l2=L.Linear(None, 1200),
			l3=L.Linear(None, 10),
		)

	def __call__(self, x, apply_softmax=True):
		out = selu(self.l1(x))
		out = selu(self.l2(out))
		out = self.l3(out)
		if apply_softmax:
			out = F.softmax(out)
		return out

class ReluBatchNormModel(chainer.Chain):
	def __init__(self):
		super(ReluBatchNormModel, self).__init__(
			l1=L.Linear(784, 1200),
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
	y1 = model(x, apply_softmax=True)
	y1.unchain_backward()
	xp = cuda.get_array_module(x)
	d = get_unit_vector(np.random.normal(size=x.shape).astype(np.float32))
	if xp is cupy:
		d = cuda.to_gpu(d)

	for i in xrange(Ip):
		d = Variable(d)
		y2 = model(x + xi * d, apply_softmax=True)
		kld = F.sum(compute_kld(y1, y2))
		kld.backward()
		d = get_unit_vector(d.grad)
	
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
	# train_data = (train_data - np.mean(train_data)) / np.std(train_data)
	# test_data = (test_data - np.mean(test_data)) / np.std(test_data)
	train_data = 2 * train_data - 1
	test_data = 2 * test_data - 1
	return (train_data, np.asanyarray(train_label, dtype=np.int32)), (test_data, np.asanyarray(test_label, dtype=np.int32))

def compute_accuracy(model, x, t, num_clusters=10):
	xp = model.xp
	if model.xp is cuda.cupy:
		x = cuda.to_gpu(x)
		t = cuda.to_gpu(t)
	with chainer.using_config("Train", False):
		probs = F.softmax(model(x, apply_softmax=True)).data
		labels_predict = xp.argmax(probs, axis=1)
		predict_counts = np.zeros((10, num_clusters), dtype=np.float32)

		for i in xrange(len(x)):
			label_predict = int(labels_predict[i])
			label_true = int(t[i])
			predict_counts[label_true][label_predict] += 1

		probs = np.transpose(predict_counts) / np.reshape(np.sum(np.transpose(predict_counts), axis=1), (num_clusters, 1))
		indices = np.argmax(probs, axis=1)
		match_count = np.zeros((10,), dtype=np.float32)
		for i in xrange(num_clusters):
			assinged_label = indices[i]
			match_count[assinged_label] += predict_counts[assinged_label][i]

		accuracy = np.sum(match_count) / len(x)
		return predict_counts.astype(np.int), accuracy

def train(args):
	try:
		os.mkdir(args.out)
	except:
		pass
	mnist_train, mnist_test = get_mnist()
		
	# init model
	model = ReluBatchNormModel()
	if args.gpu_device >= 0:
		cuda.get_device(args.gpu_device).use()
		model.to_gpu()
	xp = model.xp

	# init optimizer
	optimizer = optimizers.Adam(alpha=args.learning_rate, beta1=0.9)
	optimizer.setup(model)
	optimizer.add_hook(GradientClipping(1))

	# IMSAT hyperparameters
	lam = 0.2
	mu = 4.0

	train_data, train_label = mnist_train
	if args.gpu_device >= 0:
		train_data = cuda.to_gpu(train_data)
		train_label = cuda.to_gpu(train_label)
	train_loop = len(train_data) // args.batchsize
	train_indices = np.arange(len(train_data))

	for epoch in xrange(1, args.epoch):
		np.random.shuffle(train_indices)	# shuffle data
		sum_loss = 0

		with chainer.using_config("Train", True):
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

				hy = compute_marginal_entropy(p)
				hy_x = F.mean(compute_entropy(p))
				Rsat = -F.mean(compute_lds(model, x))

				loss = Rsat - lam * (mu * hy - hy_x)

				model.cleargrads()
				loss.backward()
				optimizer.update()

				if itr % 50 == 0:
					sys.stdout.write("\riteration {}/{}".format(itr, train_loop))
					sys.stdout.flush()
				sum_loss += float(loss.data)

		with chainer.using_config("Train", False):
			counts_train, accuracy_train = compute_accuracy(model, train_data, train_label)
			test_data, test_label = mnist_test
			counts_test, accuracy_test = compute_accuracy(model, test_data, test_label)

		sys.stdout.write("\r\033[2KEpoch {} - loss {:.5f} - acc {:.4f} (train), {:.4f} (test)\n".format(epoch, sum_loss / train_loop, accuracy_train, accuracy_test))
		sys.stdout.flush()
		print(counts_train)
		print(counts_test)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--out", "-o", type=str, default="model")
	parser.add_argument("--gpu-device", "-g", type=int, default=0)
	parser.add_argument("--learning-rate", "-lr", type=float, default=0.01)
	parser.add_argument("--epoch", "-e", type=int, default=500)
	parser.add_argument("--batchsize", "-b", type=int, default=256)
	args = parser.parse_args()
	train(args)