from itertools import combinations
from pyspark import SparkContext
from operator import add
import sys
import math
import time
import json
import random
from collections import defaultdict

random.seed(5)

def to_list(a):
    return [a]


def append(a,b):
    a.append(b)
    return a


def extend(a,b):
    a.extend(b)
    return a


def createBusinessIdUserIndex(businessId, userId, lookup):
    indices = []
    for i in userId:
        indices.append(lookup[i])
    return (businessId, indices)


def createSignatureMatrix(businessIdUserIndex,hash_dic):
    signature_matrix = businessIdUserIndex.map(lambda x: (x[0], [hash_dic[i] for i in x[1]])).map(lambda x: (x[0], [min(i) for i in zip(*x[1])])).sortByKey()
    return signature_matrix


def LSH(signatureMatrix):
    candidates = set()
    for i in range(0,150,3):
        start_index  = i
        end_index = start_index + 3
        bucket = defaultdict(set)
        for index in range(len(signatureMatrix)):
            bid = signatureMatrix[index][0]
            sign = signatureMatrix[index][1]
            hashed_value = hash(tuple(sign[start_index:end_index])) % 10000
            bucket[hashed_value].add(bid)
        
        for items in bucket.values():
            if len(items)>1:
                candidate_pairs = combinations(items,2)
                for pair in candidate_pairs:
                    candidates.add(tuple(sorted(pair)))

    return candidates


if __name__=='__main__':
    if len(sys.argv) !=3:
        print("All arguments haven't been specified")
        sys.exit(-1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    sc= SparkContext('local[*]','Task1')
    sc.setLogLevel("WARN")

    start_time = time.time()

    file = sc.textFile(input_file)
    header = file.first()
    reviews = file.filter(lambda x: x!=header).map(lambda x: x.split(",")).map(lambda x: (x[1], x[0])).distinct().persist()

    userIdIndex = reviews.map(lambda x: x[1]).distinct().zipWithIndex().persist().collectAsMap()
    userIndexList = list(userIdIndex.values())
    buckets = len(userIndexList)


    hash_dic = defaultdict(list)
    prime = 49157

    for i in range(buckets):
        for j in range(150):
            hash_val = ((random.randrange(50,100000)*i + random.randrange(50,100000)) % prime) % buckets
            hash_dic[i].append(hash_val)

    businessIdUserId = reviews.combineByKey(to_list, append, extend)
    businessIdUserIdDict = businessIdUserId.collectAsMap()
    businessIdUserIndex = businessIdUserId.map(lambda x: createBusinessIdUserIndex(x[0], x[1], userIdIndex))
    
    signatureMatrix = createSignatureMatrix(businessIdUserIndex, hash_dic)

    candidates = LSH(signatureMatrix.collect())

    results = list()
    for candidate in candidates:
        business1, business2 = candidate[0], candidate[1]
        users1, users2 = set(businessIdUserIdDict[business1]), set(businessIdUserIdDict[business2])

        jaccardValue = len(users1&users2) / len(users1|users2)
        if jaccardValue>=0.5:
            result_dic = {}
            result_dic["b1"] = business1
            result_dic["b2"] = business2
            result_dic["sim"] = jaccardValue
            results.append(result_dic)
            
    results.sort(key = lambda x:(x["b1"],x["b2"],x["sim"]))

    with open(output_file, "w") as f:
        f.write("business_id_1,business_id_2,similarity")
        f.write("\n")
        for result in results:
            f.write(str(result["b1"]))
            f.write(",")
            f.write(str(result["b2"]))
            f.write(",")
            f.write(str(result["sim"]))
            f.write("\n")

    print(time.time()-start_time)
    
    sc.stop()
    