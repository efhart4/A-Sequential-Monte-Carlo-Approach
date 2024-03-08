from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle


from Model_I_experiment_I_functions import test_MH_sampler


def worker(_):
    return test_MH_sampler()

if __name__ == '__main__':
    with Pool() as p:
        num_processes = 3  # Set this to the number of parallel processes you want to run
        results = p.map(worker, range(num_processes))


        
    #with open('/model1_0.npy', 'wb') as f:
        #pickle.dump(results, f)

 
