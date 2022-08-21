from blackbox import BlackBox
import sys
import random
import binascii
import time

NO_HASH_FUNCS = 120
PRIME = 1610612741
WINDOW_SIZE = 10
random.seed(553)
sum_ground = 0
sum_estimate = 0

previous_users = set()
results = []

a = random.sample(range(0, sys.maxsize), NO_HASH_FUNCS)
b = random.sample(range(0, sys.maxsize), NO_HASH_FUNCS)

def myhashs(s):
    hash_result = list()
    for i in range(0,NO_HASH_FUNCS):
        hash_val = ((a[i]*int(binascii.hexlify(s.encode('utf8')),16) + b[i]) % PRIME) % sys.maxsize
        hash_result.append(hash_val)
    return hash_result

def trailingZeros(x):
    return len(x) - len(x.rstrip('0'))

def Average(lst):
    return sum(lst)/len(lst)

def FM(users):
    global sum_ground
    global sum_estimate
    max_trailing_zeros = [0]*NO_HASH_FUNCS
    ground_truth = len(set(users))
    
    for user in users:
        trailZeroList = list()
        hash_result = myhashs(user)
        for hash_val in hash_result:
            trailZeroList.append(trailingZeros(bin(hash_val)[2:]))
        max_trailing_zeros = [max(i,j) for i,j in zip(max_trailing_zeros,trailZeroList)]
    
    unique_esitmates = sorted([pow(2,r) for r in max_trailing_zeros])
    avgWindowList = list()
        
    for windowRange in range(0,NO_HASH_FUNCS,WINDOW_SIZE):
        avgWindowList.append(Average(unique_esitmates[windowRange:(windowRange+WINDOW_SIZE)]))

    median_estimate = round(avgWindowList[len(avgWindowList)//2])
    results.append([ground_truth,median_estimate])
    sum_ground += ground_truth
    sum_estimate += median_estimate



if __name__=='__main__':
    if len(sys.argv) !=5:
        print("All arguments haven't been specified")
        sys.exit(-1)

    start_time = time.perf_counter()
    
    input_file = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_of_asks = int(sys.argv[3])
    output_file = sys.argv[4]

    bx = BlackBox()
    for _ in range(num_of_asks):
        stream_users = bx.ask(input_file, stream_size)
        FM(stream_users)

    print(sum_estimate/sum_ground)
    
    with open(output_file, "w") as f:
        f.write("Time,Ground Truth,Estimation\n")
        for index,result in enumerate(results):
            f.write(str(index))
            f.write(",")
            f.write(str(result[0]))
            f.write(",")
            f.write(str(result[1]))
            f.write("\n")
    
    print(time.perf_counter()-start_time)
