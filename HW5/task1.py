from blackbox import BlackBox
import sys
import random
import binascii
import time

BIT_ARRAY_LENGTH  = 69997
NO_HASH_FUNCS = 10
PRIME = 49157
random.seed(553)

bit_array = [0]*BIT_ARRAY_LENGTH
previous_users = set()
results = []

a = [74393, 95573, 59063, 22884, 23977, 59916, 81306, 96678, 74887, 3718]
b = [54571, 70788, 77887, 6497, 6412, 33490, 80663, 84366, 91948, 30540]

def myhashs(s):
    result = list()
    for i in range(0,NO_HASH_FUNCS):
        hash_val = ((a[i]*int(binascii.hexlify(s.encode('utf8')),16) + b[i]) % PRIME) % BIT_ARRAY_LENGTH
        result.append(hash_val)
    return result


def BloomFilter(users):
    false_postive, true_negative = 0.0,0.0
    seen = 0

    for user in users:
        hash_index = myhashs(user)
        for index in hash_index:
            if bit_array[index] == 1:
                seen = 1
            else:
                seen = 0
                true_negative = true_negative + 1.0
                break

        if(seen==1):
            if user not in previous_users:
                false_postive = false_postive + 1.0

        for index in hash_index:
            bit_array[index] = 1

        previous_users.add(user)
        
    if true_negative==0 and false_postive==0:
        false_postive_rate = 0.0
    else:
        false_postive_rate = false_postive/(false_postive+true_negative)
    results.append(false_postive_rate)


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
        BloomFilter(stream_users)

    with open(output_file, "w") as f:
        f.write("Time,FPR\n")
        for index,result in enumerate(results):
            f.write(str(index))
            f.write(",")
            f.write(str(result))
            f.write("\n")
    
    print(time.perf_counter()-start_time)
