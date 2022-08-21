from itertools import combinations
from pyspark import SparkContext
from operator import add
import sys
import math
import time


def to_list(a):
    return [a]

def append(a,b):
    a.append(b)
    return a

def extend(a,b):
    a.extend(b)
    return a

def createBaskets(data):
    baskets = []
    transactions = dict()
    item_set = set()
    # for record in data:
    #     user_id, business_id  = record.split(',')
    #     user_id, business_id = int(user_id), int(business_id)
    #     if user_id not in transactions.keys():
    #         transactions[user_id] = set()
    #     transactions[user_id].add(business_id)
    #     item_set.add(frozenset([business_id]))
    
    for record in data:
        user_id, business_id = record[0], record[1]
        baskets.append(business_id)
        for item in business_id:
            item_set.add(frozenset([item]))
    
    # for record in transactions.values():
    #     baskets.append(record)
    
    return item_set, baskets

def createLSets(candidateSet, baskets, itemSupport):
    frequentSets = set()
    temp = {}

    for item in candidateSet:
        for record in baskets:
            if item.issubset(record):
                if item not in temp.keys():
                    temp[item] = 1
                else:
                    temp[item] +=1
                    if(temp[item]>=itemSupport):
                        frequentSets.add(item)
                        break
    return frequentSets

def apriori(data,actualSupport,rddLength):
    frequentItemSet = []
    data = list(data)
    #print("Partition Length: "+str(len(data)))
    CSets, baskets  = createBaskets(data)
    #print(CSet, baskets)
    support = math.ceil(actualSupport * len(data) / rddLength)
    #print(support)
    LSets = createLSets(CSets, baskets, support)
    #print(LSet)

    sizeOfLset = 1

    while(LSets):
        frequentItemSet.append(LSets)

        CSets = set()
        for i in LSets:
            for j in LSets:
                if len(i.union(j))==(sizeOfLset+1):
                    CSets.add(i.union(j))

        removeSets = set()
        #print(LSets)
        for CSet in CSets:
            subsets = combinations(CSet, sizeOfLset)
            for subset in subsets:
                #print(subset)
                if frozenset(subset) not in LSets:
                    removeSets.add(CSet)
                    break
        prunedCSets = CSets-removeSets
        #print(prunedCSets)

        LSets = createLSets(prunedCSets,baskets,support)
        #print(LSets)
        sizeOfLset +=1

    return(frequentItemSet)

def verifyCandidates(data, candidates):
    data = list(data)
    candidate_support = {}
    for candidate in candidates:
        candidate_support[tuple(candidate)] = 0
        for transaction in data:
            basket = transaction[1]
            if set(candidate).issubset(set(basket)):
                candidate_support[tuple(candidate)]+=1
    #print(candidate_support)
    candidate_frequency = [(key,value) for key,value in candidate_support.items()]
    return candidate_frequency

def sortedPrinting(data):
    returnString = ""
    stringLength = 1
    firstElement = True
    for i in data:

        if len(i)>stringLength:
            stringLength+=1
            returnString = returnString + '\n\n'
            firstElement = True
        if len(i)==1:
            element = "('" + tuple(i)[0] + "')"
        else:
            element = str(tuple(i))
        if firstElement==True:
            returnString = returnString + element
            firstElement = False
        else:
            returnString = returnString + ',' + element
    return returnString

if __name__=='__main__':
    if len(sys.argv) !=5:
        print("All arguments haven't been specified")
        sys.exit(-1)
    
    case_number = int(sys.argv[1])
    support = int(sys.argv[2])
    input_file = sys.argv[3]
    output_file = sys.argv[4]
    
    sc= SparkContext('local[*]','Task1')
    sc.setLogLevel("WARN")

    start_time = time.time()
    lines = sc.textFile(input_file)
    header = lines.first()

    if(case_number==1):
        lines = lines.filter(lambda x: x!=header).map(lambda x: x.split(",")).combineByKey(to_list, append, extend).persist()
    else:
        lines = lines.filter(lambda x: x!=header).map(lambda x: x.split(",")).map(lambda x: (x[1], x[0])).combineByKey(to_list, append, extend).persist()
    
    rddLength = lines.count()

    #print("RDD Length: "+str(rddLength))
    #print(lines.collect())

    candidates = lines.mapPartitions(lambda x: apriori(x, support, rddLength)).flatMap(lambda x: x).map(lambda x: (x,1)).reduceByKey(add).map(lambda x: x[0]).collect()
    candidates = list(map(list, candidates))
    #candidates = lines.mapPartitions(lambda x: apriori(x, support, rddLength)).collect()
    candidates.sort(key = lambda x: (len(x),x.sort(),x))
    finalCandidates = sortedPrinting(candidates)

    frequent = lines.mapPartitions(lambda x: verifyCandidates(x, candidates)).reduceByKey(add).filter(lambda x: x[1]>=support).map(lambda x: x[0]).collect()
    frequent = list(map(list, frequent))
    #print(frequent)
    frequent.sort(key = lambda x: (len(x),x.sort(),x))
    finalFrequentSets = sortedPrinting(frequent)

    with open(output_file, "w") as f:
        f.write("Candidates:\n")
        f.write(finalCandidates)
        f.write("\n\n")
        f.write("Frequent Itemsets:\n")
        f.write(finalFrequentSets)
    total_time = time.time() - start_time
    print("Duration: "+str(total_time))
    sc.stop()