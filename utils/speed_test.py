import numpy as np
import datetime
import time
from numba import jit

def measureTime(name, func):
    start = datetime.datetime.now()
    v = func()
    end = datetime.datetime.now()

    print(f'{name} took: {(end-start).microseconds} us ({(end-start).seconds}s)')
    return v

#@jit(nopython=True)
def method1(mat):
    mat[:,2] += 5
    return mat

@jit(nopython=True)
def method2(mat):
    mat += np.array([0,0,5], dtype=np.float32)
    return mat

# Create a big and random array
a = measureTime('Random init', lambda: np.random.rand(1000, 3))
print(a.shape)
print(a[0])

# Add height value
height = 5

num_tries = 1000
total_duration = datetime.timedelta(microseconds=0)
for i in range(num_tries+1):
    start = datetime.datetime.now()
    v = method1(a)
    end = datetime.datetime.now()

    duration = end-start
    if i > 0:
        total_duration += duration
    print(f'took: {duration.microseconds} us ({duration.seconds}s)')

print(f'Total time: {total_duration.seconds} s + {total_duration.microseconds} us')
avg = (total_duration.seconds / num_tries) * 1e6 + total_duration.microseconds / num_tries
print(f'Average time: {avg} us')

# method1(a)
# b = measureTime('Method 1', lambda: method1(a))
# print(b[0])
# method2(a)
# c = measureTime('Method 2', lambda: method2(a))
# print(c[0])

