import numpy as np
import time


np.random.seed(960501)


def generate_conditions(N, d):
    print('Generating... ')
    env = 3. * np.random.randn(d, 1).astype('f')             # environment
    X = 9. * np.random.randn(N, d).astype('f')               # observations
    y = np.dot(X, env) + np.random.randn(N, 1).astype('f')   # target
    print('Completed')
    return X, y

N, d = int(1e5), int(1e5)
it, realit, niter = 0, 0, int(1e5)
h = 1e-2

X, y = generate_conditions(N, d)
theta = np.random.randn(d, 1).astype('f') * 3.  # initial theta

F = lambda theta: np.linalg.norm((np.dot(X, theta) - y) / N)

t0 = time.time()
F0 = F(theta)
F1 = F0
while True:
    theta1 = theta - (h * 2.) * np.dot(X.T, np.dot(X, theta) - y)
    F1 = F(theta1)
    if F1 > F0:
        h /= 2.
    else:
        if it % 10 == 9:
            print(f'\rStep {it + 1:7d}, F0: {F0:10.6f}, F1: {F1:10.6f}, crit: {1. - F1 / F0:10.9f}')
        it += 1
        h *= 1.2
        theta = theta1
        F0 = F1
        if F0 < 1.:
            break
    realit += 1
dt = time.time() - t0
print(f'it: {it:4d}, rmse: {F0:10.5f}, t: {dt:8.5f} s, time/iter: {dt/realit*1000:9.4f} ms')
