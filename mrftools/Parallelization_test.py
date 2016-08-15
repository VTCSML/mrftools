from joblib import Parallel, delayed
import multiprocessing
import time
import xlwt
from xlwt import Workbook


inputs = range(3)
def processInput(i):
    wb = Workbook()
    sheet1 = wb.add_sheet('Results')
    sheet1.write(0,0,'Max_iter')
    wb.save(str(i) + 'result.xls')
    time.sleep(2)



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

