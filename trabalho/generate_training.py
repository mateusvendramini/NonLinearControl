from system_sim import Sistema
import os
import numpy as np 
import itertools
import multiprocessing
from multiprocessing import Process, freeze_support
import warnings
def map_random(min, max, rand):
    return min + (max-min)*rand

# def input_map(input, rng):
#     return  input[0] + map_random(-0.1, 0.1, rng.random()), input[1] + map_random(-0.1, 0.1, rng.random()), input[2] + map_random(-0.1, 0.1, rng.random()), input[3] + map_random(-0.1, 0.1, rng.random()), input[4] + map_random(-0.1, 0.1, rng.random()), input[5] + map_random(-0.1, 0.1, rng.random()), input[6] + map_random(-0.1, 0.1, rng.random()), input[7] + map_random(-0.1, 0.1, rng.random()), input[8] + map_random(-0.1, 0.1, rng.random()), input[9] + map_random(-0.1, 0.1, rng.random()), input[10] + map_random(-0.1, 0.1, rng.random()), \
#         input[11] + map_random(-0.1, 0.1, rng.random()), input[12] + map_random(-0.1, 0.1, rng.random()), input[13] + map_random(-0.1, 0.1, rng.random()), input[14] + map_random(-0.1, 0.1, rng.random()), input[15] + map_random(-0.1, 0.1, rng.random()), input[16] + map_random(-0.1, 0.1, rng.random()), input[17] + map_random(-0.1, 0.1, rng.random()), input[18] + map_random(-0.1, 0.1, rng.random()), input[19] + map_random(-0.1, 0.1, rng.random()), input[20] + map_random(-5, 5, rng.random()), input[21] + map_random(-5, 5, rng.random())

def process(inputs, i):
    s = 666 + i
    rng = np.random.default_rng(seed=s)
    Y = np.empty((0,8))  
    X = np.empty((0,16))
    j=0
    j_max=len(inputs)
    j_notify = j_max/100
    for input in inputs:
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            sys = Sistema(input[0] + map_random(-0.1, 0.1, rng.random()), input[1] + map_random(-0.1, 0.1, rng.random()), input[2] + map_random(-0.1, 0.1, rng.random()), input[3] + map_random(-0.1, 0.1, rng.random()), input[4] + map_random(-0.1, 0.1, rng.random()), input[5] + map_random(-0.1, 0.1, rng.random()), input[6] + map_random(-0.1, 0.1, rng.random()), input[7] + map_random(-0.1, 0.1, rng.random()), input[8] + map_random(-0.01, 0.01, rng.random()), input[9] + map_random(-0.01, 0.01, rng.random()), input[10] + map_random(-0.01, 0.01, rng.random()), \
                input[11] + map_random(-0.01, 0.01, rng.random()), input[12] + map_random(-0.1, 0.1, rng.random()), input[13] + map_random(-0.1, 0.1, rng.random()), input[14] + map_random(-0.1, 0.1, rng.random()), input[15] + map_random(-0.1, 0.1, rng.random()), input[16] + map_random(-0.1, 0.1, rng.random()), input[17] + map_random(-0.1, 0.1, rng.random()), input[18] + map_random(-0.1, 0.1, rng.random()), input[19] + map_random(-0.1, 0.1, rng.random()), input[20], input[21])
            try:
                sys.run()
                x, y = sys.getTrainingArray()
                X = np.concatenate((X, x))
                Y = np.concatenate((Y, y))
            except Warning:
                print("Error processing output for {0}, {1}, {2},{3},{4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}, {13}".format(sys.m1, sys.m1h, sys.m2, sys.m2h, sys.L1, sys.L1h, sys.L2, sys.L2h, sys.I1, sys.I1h, sys.I2, sys.I2h, sys.q10, sys.q20))
                print("Continuing process {0}".format(i))
                continue
            finally:
                j+=1
                if (j % j_notify == 0):
                    print("Andamento {0} \%".format(j/j_notify))



    x_file_name='x_{0}.out'.format(i)
    y_file_name='y_{0}.out'.format(i)
    np.save(os.path.join('.', 'np_arrays', x_file_name), X)
    np.save(os.path.join('.', 'np_arrays', y_file_name), Y)

def main():
    freeze_support()
    param_len = 2 # um dia 10
    m1 = np.linspace(2, 8, param_len) #0 m1
    m1h = np.linspace(2, 8, param_len)
    m2 = np.linspace(1, 5, param_len) #2 m2
    m2h = np.linspace(1, 5, param_len)
    L1 = np.linspace(1, 2, param_len) #4 L1 
    L1h = np.linspace(1, 2, param_len)
    L2 = np.linspace(0.5, 1.5, param_len) #6 L2 
    L2h = np.linspace(0.5, 1.5, param_len)
    I1 = np.linspace(0.1, 0.4, param_len) # 8 I1
    I1h = np.linspace(0.1, 0.4, param_len)
    I2 = np.linspace(0.05, 0.2, param_len) # 10 I2
    I2h = np.linspace(0.05, 0.2, param_len)
    F1 = np.linspace(10, 20, param_len) # 12
    F1h = np.linspace(10, 20, param_len)
    F2 = np.linspace(10, 20, param_len) #14
    F2h = np.linspace(10, 20, param_len)
    q10 = np.linspace(np.pi/3, 5*np.pi/3, 3)
    q20 = np.linspace(0, 2*np.pi, 3)
    q1jump = np.linspace(-np.pi/3, np.pi/3, 3)
    q2jump = np.linspace(-np.pi/2, np.pi/2, 3)
    K1 = np.linspace(74.00, 74.00, 1)
    K2 = (266.00, 266.00, 1)
    # m1, m1h, m2, m2h, L1, L1h, L2, L2h, I1, I1h, I2, I2h, F1, F1h, F2, F2h, ref1, ref2, q10, q20, K1, K2):

    inputs = itertools.product(m1, m1h, m2, m2h, L1, L1h, L2, L2h, I1, I1h, I2, I2h, F1, F1h, F2, F2h, q1jump, q2jump, q10, q20, K1, K2)
    #rng = np.random.default_rng(seed=666)

    input_list = list(map(list, inputs))
    print('input_len ' + str(len(input_list)))
    input_list = input_list[0:len(input_list)//128]
    #print(len(input_list))

    process_number = multiprocessing.cpu_count()/2
    slices = []
    input_len = len (input_list)
    slice_size = input_len // process_number
    for i in range(process_number):
        slice_i = input_list[i*slice_size:(i+1)*slice_size]
        slices.append(slice_i)

    procs = []
    i = 0
    for slice in slices:
        #print(slice[1])
        proc = Process(target=process, args=(slice, i,))
        procs.append(proc)
        proc.start()
        #break
        i+= 1

    for proc in procs:
        proc.join()

if __name__ == '__main__':
    main()
# print(len(slices))
# print(len(slices[0]))
#print(np.shape(input_list))

# computador com 16 threads para calcular os diversos modelos
#parametros para variavar angulos de inicio, degrau, L1, L2, I1, I2, m1, m2, 

# rng.random( gera um numero aleatório entre 0 e 1