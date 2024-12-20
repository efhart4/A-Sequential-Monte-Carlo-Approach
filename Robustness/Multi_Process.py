from multiprocessing import Pool
import pickle


from Distribution_experiment import test_MH_sampler


def worker(_):
    return test_MH_sampler()

if __name__ == '__main__':
    with Pool() as p:
        num_processes = 3  # Set this to the number of parallel processes you want to run
        results = p.map(worker, range(num_processes))


        
    with open('/Users/modelt/Documents/Research/Extended Particle FIlter/finished code/Robustness 2/results.npy', 'wb') as f:
        pickle.dump(results, f)

 
