import random
import numpy as np
import time

start = time.time()
x = np.random.random(10**8)
end = time.time()

print("花费的时间是：{}".format(end - start))