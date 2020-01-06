import numpy as np
from numba import cuda, jit, float32
import math


threadsperblock = (16, 16)
dtype = np.float32

################################################################
######################### for OLS & GD #########################
################################################################

def ols(X, y_):
    theta_lstsq, err_lstsq, _, _ = np.linalg.lstsq(X, y_, rcond=None)
    return theta_lstsq, np.sqrt(err_lstsq)

def gd(X, y_, h_=1e-2, niter=30):
    theta = np.random.randn(X.shape[1], 1)  # initial theta
    h = h_
    F = lambda theta: np.linalg.norm(np.dot(X, theta) - y_)

    XTX = np.dot(X.T, X)
    XTy = np.dot(X.T, y_)

    F0 = F(theta)
    for i in range(niter):
        theta1 = theta - (h * 2.) * (np.dot(XTX, theta) - XTy)
        F1 = F(theta1)
        if F1 > F0:
            h /= 2.
        else:
            h *= 1.2
            theta = theta1
            F0 = F1
    
    return theta, F0

################################################################
######################## for numba cuda ########################
################################################################

def get_bpg(matrix):
    x, y = matrix if type(matrix) == tuple else matrix.shape
    blockspergrid_X = round(.5 + x / threadsperblock[0])
    blockspergrid_Y = round(.5 + y / threadsperblock[1])
    return blockspergrid_X, blockspergrid_Y

@cuda.jit
def dev_matmul(A, B, C):
    x, y = cuda.grid(2)
    if x < C.shape[0] and y < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[x, k] * B[k, y]
        C[x, y] = tmp
    cuda.syncthreads()

def host_matmul(A, B) -> cuda.cudadrv.devicearray.DeviceNDArray:
    res_shape = A.shape[0], B.shape[1]
    blockspergrid = get_bpg(res_shape)
    C = cuda.device_array(res_shape, dtype=dtype)
    dev_matmul[blockspergrid, threadsperblock](A, B, C)
    return C

# @cuda.jit
# def dev_dot(X, theta, y):
#     x, _ = cuda.grid(2)
#     if x < X.shape[0]:
#         tmp = 0.
#         for k in range(X.shape[1]):
#             tmp += X[x, k] * theta[k, 0]
#         y[x, 0] = tmp
#     cuda.syncthreads()

@cuda.jit
def dev_dot(X, theta, y):
    cache = cuda.shared.array(shape=threadsperblock, dtype=float32)
    m, n = cuda.grid(2)
    stride = X.shape[1] // threadsperblock[0]
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    if m < X.shape[0] and n < X.shape[1]:
        tmp = 0.
        idx = 0
        for i in range(stride):
            idx = n * stride + i
            if idx < X.shape[1]:  # frequent condition evaluation may cause overhead
                tmp += X[m, idx] * theta[idx, 0]
            cache[tx, ty] = tmp
            cuda.syncthreads()
        i = cuda.blockDim.x // 2;
        while i != 0:
            if ty < i:
                cache[tx, ty] += cache[tx, ty + i];
            cuda.syncthreads();
            i //= 2;
        y[m, 0] = cache[tx, 0]
    cuda.syncthreads()

def host_dot(X, theta):
    res_shape = X.shape[0], 1
    blockspergrid = get_bpg(res_shape)
    y = cuda.device_array(res_shape, dtype=dtype)
    dev_dot[blockspergrid, threadsperblock](X, theta, y)
    return y

@cuda.jit
def dev_sub(A, B, C):
    x, y = cuda.grid(2)  # threadIdx.[] + blockIdx.[] * blockDim.[]
    if x < A.shape[0] and y < A.shape[1]:
        C[x, y] = A[x, y] - B[x, y]
    cuda.syncthreads()

def host_sub(self, other):
    blockspergrid = get_bpg(self)
    dev_C = cuda.device_array((self.shape), dtype=dtype)
    dev_sub[blockspergrid, threadsperblock](self, other, dev_C)
    return dev_C

@cuda.jit
def dev_rmul_number_literal(self, other, res):
    x, y = cuda.grid(2)
    if x < self.shape[0] and y < self.shape[1]:
        res[x, y] = other * self[x, y]
    cuda.syncthreads()

def host_rmul(self, other):
    if type(other) in [float, int]:
        blockspergrid = get_bpg(self)
        dev_C = cuda.device_array((self.shape), dtype=dtype)
        dev_rmul_number_literal[blockspergrid, threadsperblock](self, other, dev_C)
        return dev_C

cuda.cudadrv.devicearray.DeviceNDArray.__sub__ = host_sub  # overload
cuda.cudadrv.devicearray.DeviceNDArray.__rmul__ = host_rmul  # overload

def cudagd(X, y_, h_=1e-2, niter=30):
    h = h_
    theta = np.random.randn(X.shape[1], 1)  # initial theta

    # to device
    dev_X = cuda.to_device(X.astype(dtype))
    dev_y_ = cuda.to_device(y_.astype(dtype))
    dev_theta = cuda.to_device(theta.astype(dtype))

    dev_XTX = cuda.to_device(np.dot(X.T, X).astype(dtype))
    dev_XTy = cuda.to_device(np.dot(X.T, y_).astype(dtype))

    F = lambda dev_theta: np.linalg.norm((host_dot(dev_X, dev_theta) - dev_y_).copy_to_host())
    F0 = F(dev_theta)
    for i in range(niter):
        dev_theta1 = dev_theta - (h * 2.) * (host_dot(dev_XTX, dev_theta) - dev_XTy)
        F1 = F(dev_theta1)
        if F1 > F0:
            h /= 2.
        else:
            h *= 1.2
            dev_theta = dev_theta1  # check overhead
            F0 = F1
    
    return dev_theta.copy_to_host(), F0

################################################################
