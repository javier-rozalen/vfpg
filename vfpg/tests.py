# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 13:23:51 2022

@author: javir
"""

#%%
import multiprocessing,time

def square(n):
   result = n*n
   time.sleep(5)
   return result
if __name__ == '__main__':
   inputs = list(range(5))
   p = multiprocessing.Pool(processes = multiprocessing.cpu_count())
   p_outputs = p.map(square, inputs)
   p.close()
   p.join()
   print ('Pool :', p_outputs)
    
#%%
import time
from multiprocessing import Process


def cube(x):
    print(f"start process {x}")
    print(x * x * x)
    time.sleep(3)
    print(f"end process {x}")


if __name__ == "__main__":
    processes = []
    t_i = time.time()
    for i in range(5):
        p = Process(target=cube, args=(i,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print(f'Total time: {time.time()-t_i}')
    
    
#%% 
import time
from multiprocessing import Pool
import multiprocessing as mp

def cube(x):
    print(f"start process {x}")
    result = x * x * x
    time.sleep(1)
    print(f"end process {x}")
    return result

n_cores = mp.cpu_count()
if __name__ == "__main__":
    ts = time.time()
    pool = Pool(processes=n_cores)
    print([pool.apply(cube, args=(x,)) for x in range(10)])
    pool.close()
    pool.join()
    print("Time in parallel:", time.time() - ts)