import chainer, math
import numpy as np
from chainer import cuda
from chainer import function
from chainer.utils import type_check

class SELU(function.Function):
	def __init__(self, alpha, lam):
		self.alpha = float(alpha)
		self.lam = float(lam)

	def check_type_forward(self, in_types):
		type_check.expect(in_types.size() == 1)
		x_type, = in_types
		type_check.expect(x_type.dtype.kind == 'f')

	def forward_cpu(self, x):
		y = x[0].copy()
		neg_indices = x[0] <= 0
		y[neg_indices] = self.alpha * (np.exp(y[neg_indices]) - 1)
		y *= self.lam
		return y,

	def forward_gpu(self, x):
		y = cuda.elementwise(
			'T x, T alpha, T lam', 'T y',
			'y = x > 0 ? (T)(lam * x) : (T)(lam * alpha * (exp(x) - 1))',
			'elu_fwd')(x[0], self.alpha, self.lam)
		return y,

	def backward_cpu(self, x, gy):
		gx = gy[0].copy()
		neg_indices = x[0] <= 0
		gx[neg_indices] *= self.alpha * np.exp(x[0][neg_indices])
		gx *= self.lam
		return gx,

	def backward_gpu(self, x, gy):
		gx = cuda.elementwise(
			'T x, T gy, T alpha, T lam', 'T gx',
			'gx = x > 0 ? (T)(lam * gy) : (T)(lam * gy * alpha * exp(x))',
			'elu_bwd')(
				x[0], gy[0], self.alpha, self.lam)
		return gx,


def selu(x, alpha=1.6732632423543772848170429916717, lam=1.0507009873554804934193349852946):
	return SELU(alpha, lam)(x)

def dropout_selu(x, ratio=0.1, alpha=-1.7580993408473766):
	if chainer.config.train == False:
		return x

	q = 1.0 - ratio

	xp = cuda.get_array_module(*x)
	if xp == np:
		d = np.random.rand(*x[0].shape) >= ratio
	else:
		d = xp.random.rand(*x[0].shape, dtype=np.float32) >= ratio

	a = math.pow(q + alpha ** 2 * q * (1 - q), -0.5)
	b = -a * (1 - q) * alpha

	return a * (x * d + alpha * (1 - d)) + b
