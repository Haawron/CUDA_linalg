from modules import *
import time
from typing import Tuple


def generate_conditions(N, d) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(960501)
    env = np.sin(np.random.random((d, 1)) * 2 * np.pi) * 3 - 1.5  # environment
    X = np.random.randn(N, d)  # observations
    y_ = np.where(np.dot(X, env) > 0, 1, 0)  # target
    return X, y_

data = []
for N in 10 ** np.arange(3, 6):
    for d in 10 ** np.arange(2, 5):
        print(f'Beginning N: {N}, d: {d}')
        X, y_ = generate_conditions(N, d)
        t0 = time.time()
        if N * d < 1e8:
            theta1, err1 = ols(X, y_); t1 = time.time()
            print(f'N: {N:6}, d: {d:5}, t: {t1-t0:9.4f}s, err: {err1[0] if err1 else 0.0:9.5f}\tOLS completed')
        else:
            print('*' * 50 + ' ' * 6 + 'passed OLS')
        theta2, err2 = gd(X, y_, niter=50); t2 = time.time()
        print(f'N: {N:6}, d: {d:5}, t: {t2-t1:9.4f}s, err: {err2:9.5f}\tGD completed')
        theta3, err3 = cudagd(X, y_, niter=50); t3 = time.time()
        print(f'N: {N:6}, d: {d:5}, t: {t3-t2:9.4f}s, err: {err3:9.5f}\tCUDA GD completed')
        
        data += [{
            'N': N, 'd': d,
            'error': (err1, err2, err3),
            'time': (t1 - t0, t2 - t1, t3 - t2)}]
        
        print()

print(data)
