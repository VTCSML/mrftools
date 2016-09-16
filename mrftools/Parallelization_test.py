from joblib import Parallel, delayed
import multiprocessing
import time


inputs = range(3)
def processInput(i):
    time.sleep(5)
    return i * i



start1 = time.time()
num_cores = multiprocessing.cpu_count()
results = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in inputs)
elapsed1 = time.time() - start1
print ("time elapsed for parallel: %f" % elapsed1)

start2 = time.time()
for i in range(3):
    result = processInput(i)
elapsed2 = time.time() - start2
print ("time elapsed for sequential: %f" % elapsed2)

