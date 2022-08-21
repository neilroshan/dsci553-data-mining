from blackbox import BlackBox
import sys
import random
import time

random.seed(553)
users_list = []
results = []
seq_number = 0
first_100 = True

def fixedSizeSampling(users):
    global users_list
    global results
    global seq_number
    global first_100
    if(first_100):
        users_list = users
        first_100=False
        seq_number+=100
    else:
        for user in users:
            seq_number+=1
            p = 100.0/seq_number
            if random.random()<p:
                pop_pos = random.randint(0,stream_size-1)
                users_list.pop(pop_pos)
                users_list.insert(pop_pos,user)
    results.append([seq_number,users_list[0],users_list[20],users_list[40],users_list[60],users_list[80]])

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
        fixedSizeSampling(stream_users)

    with open(output_file, "w") as f:
        f.write("seqnum,0_id,20_id,40_id,60_id,80_id\n")
        for i in range(len(results)):
            f.write(str(results[i][0])+","+str(results[i][1])+","+str(results[i][2])+","+str(results[i][3])+","+str(results[i][4])+","+str(results[i][5]))
            f.write("\n")
    
    print(time.perf_counter()-start_time)