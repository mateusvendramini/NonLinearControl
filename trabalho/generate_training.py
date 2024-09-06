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
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    if iteration >= total: 
        print()
    
def process(inputs, i):
    s = 666 + i
    rng = np.random.default_rng(seed=s)
    Y = np.empty((0,8))  
    X = np.empty((0,16))
    j=0
    j_max=len(inputs)
    j_notify = j_max//10
    log_fn = '{0}.log'.format(i)
    log = os.path.join('.', 'logs2', log_fn)
    log_file = os.open(log, os.O_RDWR | os.O_CREAT)
    if (i==0):
        printProgressBar(j, j_max)
        #print("j_max {0} elements in j_notify chuncks {1}".format(j_max, j_notify))
    for input in inputs:
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            err1 = input[0] + map_random(-0.1, 0.1, rng.random())
            err2 = input[1] + map_random(-0.1, 0.1, rng.random())
            sys = Sistema(5 + map_random(-3, 3, rng.random()), 5 + map_random(-3, 3, rng.random()), #m1
                          3 + map_random(-2, 2, rng.random()), 3 + map_random(-2, 2, rng.random()), #m2
                            1.5 + map_random(-0.5, 0.5, rng.random()), 1.5 + map_random(-0.5, 0.5, rng.random()), # L1
                              1 + map_random(-0.5, 0.5, rng.random()), 1 + map_random(-0.5, 0.5, rng.random()),  #L2
                              0.25 + map_random(-0.15, 0.15, rng.random()), 0.25 + map_random(-0.15, 0.15, rng.random()), # I1
                              0.15 + map_random(-0.075, 0.075, rng.random()), 0.15 + map_random(-0.075, 0.075, rng.random()), #I2
                15 + map_random(-5, 5, rng.random()), 15 + map_random(-5, 5, rng.random()), #F1
                15 + map_random(-5, 5, rng.random()), 15 + map_random(-5, 5, rng.random()), #F2
                err1, err2, #ref1, ref2
                input[0] + map_random(-0.1, 0.1, rng.random()), input[1] + map_random(-0.1, 0.1, rng.random()), #q10, q20
                74, 266, #K1, K2
                map_random(0, err1*2, rng.random()), map_random(0, err2*2, rng.random())) 
            try:
                sys.run()
                x, y = sys.getTrainingArray()
                X = np.concatenate((X, x))
                Y = np.concatenate((Y, y))
            except Warning:
                line = "\r\nError processing output for [m1={0}, m1h={1}, m2={2},m2h={3},L1={4}, L1h={5}, L2={6}, L2h={7}, I1={8}, I1h={9}, I2={10}, I2h={11}, q10={12}, q20{13}, F1={14}, F1h={15}, F2={16}, F2h={17}, ref1={18}, ref2={19}]".format(sys.m1, sys.m1h, sys.m2, sys.m2h, sys.L1, sys.L1h, sys.L2, sys.L2h, sys.I1, sys.I1h, sys.I2, sys.I2h, sys.q10, sys.q20, sys.F1, sys.F1h, sys.F2, sys.F2h, sys.ref1, sys.ref2)
                os.write(log_file, str.encode(line))
                #print("Continuing process {0}".format(i))
                continue
            finally:
                j+=1
                if (i == 0  and j % j_notify == 0):
                    printProgressBar(j, j_max)
                    #print("\r\nAndamento {0}/{1}".format(j, j_max))



    x_file_name='x_{0}.out'.format(i)
    y_file_name='y_{0}.out'.format(i)
    np.save(os.path.join('.', 'np_arrays2', x_file_name), X)
    np.save(os.path.join('.', 'np_arrays2', y_file_name), Y)
    os.close(log_file)

def main():
    freeze_support()
    #param_len = 1 # um dia 10
    #m1 = np.linspace(5, 5, param_len) #0 m1 err 3
    #m1h = np.linspace(5, 5, param_len)
    #m2 = np.linspace(3, 3, param_len) #2 m2 err 2
    #m2h = np.linspace(3, 3, param_len)
    #L1 = np.linspace(1.5, 1.5, param_len) #4 L1 err 0.5
    #L1h = np.linspace(1.5, 1.5, param_len)
    #L2 = np.linspace(1, 1, param_len) #6 L2 err 0.5
    #L2h = np.linspace(1, 1, param_len)
    #I1 = np.linspace(0.25, 0.25, param_len) # 8 I1 err 0.15
    #I1h = np.linspace(0.25, 0.25, param_len)
    #I2 = np.linspace(0.125, 0.125, param_len) # 10 I2 err 0.12
    #I2h = np.linspace(0.125, 0.125, param_len)
    #F1 = np.linspace(15, 25, param_len) # 12 5
    #F1h = np.linspace(15, 15, param_len)
    #F2 = np.linspace(15, 15, param_len) #14 5
    #F2h = np.linspace(15, 15, param_len)
    q10 = np.linspace(np.pi/3, 5*np.pi/3, 16)
    q20 = np.linspace(0, 2*np.pi, 16)
    q1jump = np.linspace(-np.pi/3, np.pi/3, 8)
    q2jump = np.linspace(-np.pi/2, np.pi/2, 8)
    #K1 = np.linspace(74.00, 74.00, 1)
    #K2 = (266.00, 266.00, 1)
    # m1, m1h, m2, m2h, L1, L1h, L2, L2h, I1, I1h, I2, I2h, F1, F1h, F2, F2h, ref1, ref2, q10, q20, K1, K2):

    #inputs = itertools.product(m1, m1h, m2, m2h, L1, L1h, L2, L2h, I1, I1h, I2, I2h, F1, F1h, F2, F2h, q1jump, q2jump, q10, q20, K1, K2)
    inputs = itertools.product(q10, q20, q1jump, q2jump)
    #rng = np.random.default_rng(seed=666)

    input_list = list(map(list, inputs))
    input_len = len (input_list)
    print('input_len ' + str(len(input_list)))
    #input_list = input_list[0:input_len//1024]
    #print(len(input_list))

    process_number = multiprocessing.cpu_count()//2
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
    
    print('\r\nEnd')

if __name__ == '__main__':
    main()
# print(len(slices))
# print(len(slices[0]))
#print(np.shape(input_list))

# computador com 16 threads para calcular os diversos modelos
#parametros para variavar angulos de inicio, degrau, L1, L2, I1, I2, m1, m2, 

# rng.random( gera um numero aleatório entre 0 e 1