from itertools import combinations
from pyspark import SparkContext
import sys
import math
from operator import add
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
    item_set = set()
    
    for record in data:
        business_id = record[1]
        baskets.append(business_id)
        for item in business_id:
            item_set.add(frozenset([item]))
    
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
                unionSet = i.union(j)
                if len(unionSet)==(sizeOfLset+1):
                    CSets.add(unionSet)

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
            element = "('" + str(tuple(i)[0]) + "')"
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
    
    filter_threshold = int(sys.argv[1])
    support = int(sys.argv[2])
    input_file = sys.argv[3]
    output_file = sys.argv[4]
    
    sc = SparkContext('local[*]','Task2')
    sc.setLogLevel("WARN")

    rawData = sc.textFile(input_file)
    header = rawData.first()
    rawData = rawData.filter(lambda x: x!=header).map(lambda x: x.split(",")).map(lambda x:(str(x[0].split("\"")[1])+"_"+str(x[1].split("\"")[1]),int(x[5].split("\"")[1]))).persist()
    #print(lines.collect())
    #print(header)
    rawData = rawData.collect()
    columns = ['DATE-CUSTOMER_ID','PRODUCT_ID']

    with open('customer_product.csv', 'w') as f: 
        f.write("DATE-CUSTOMER_ID,PRODUCT_ID\n")
        for record in rawData:
            f.write(record[0])
            f.write(",")
            f.write(str(record[1]))
            f.write("\n")

    start_time = time.time()
    lines = sc.textFile("customer_product.csv")
    header = lines.first()
    lines = lines.filter(lambda x: x!=header).map(lambda x: x.split(",")).map(lambda x:(x[0],x[1])).combineByKey(to_list, append, extend).filter(lambda x: len(x[1])>filter_threshold).persist()
    rddLength = lines.count()
    print("Filtered CSV Length: "+ str(rddLength))
    #print("RDD Length: "+str(rddLength))
    #print(lines.collect())
    print("Time to get cleaned CSV: "+str(time.time()-start_time))
    candidates = lines.mapPartitions(lambda x: apriori(x, support, rddLength)).flatMap(lambda x: x).distinct().collect()
    print("Time to find Candidates after Apriori: "+ str(time.time()-start_time))
    candidates = list(map(list, candidates))
    #candidates = lines.mapPartitions(lambda x: apriori(x, support, rddLength)).collect()
    candidates.sort(key = lambda x: (len(x),x.sort(),x))
    finalCandidates = sortedPrinting(candidates)
    print("Time to find Sorted Candidates: "+ str(time.time()-start_time))

    frequent = lines.mapPartitions(lambda x: verifyCandidates(x, candidates)).reduceByKey(add).filter(lambda x: x[1]>=support).map(lambda x: x[0]).collect()
    print("Time to find frequent after secondPass: "+ str(time.time()-start_time))
    frequent = list(map(list, frequent))
    #print(frequent)
    frequent.sort(key = lambda x: (len(x),x.sort(),x))
    finalFrequentSets = sortedPrinting(frequent)
    print("Time to find Frequent Sets: "+ str(time.time()-start_time))

    with open(output_file, "w") as f:
        f.write("Candidates:\n")
        f.write(finalCandidates)
        f.write("\n\n")
        f.write("Frequent Itemsets:\n")
        f.write(finalFrequentSets)
    total_time = time.time() - start_time
    print("Duration: "+str(total_time))
    sc.stop()