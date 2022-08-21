from itertools import combinations
import string
from collections import OrderedDict
from pyspark import SparkContext
from operator import add
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
    baskets = dict(data)

    duplicate_singles = [x for i in baskets.values() for x in i]
    cleaned_baskets = [sorted(list(set(i))) for i in baskets.values()]
    item_set = set(duplicate_singles)
    
    return item_set, cleaned_baskets
    
def createLSets(candidateSet, baskets, itemSupport):
    frequentSets = set()
    temp = {}

    for item in candidateSet:
        for record in baskets:
            if set(item).issubset(record):
                if item not in temp.keys():
                    temp[item] = 1
                else:
                    temp[item] +=1
                    if(temp[item]>=itemSupport):
                        frequentSets.add(frozenset(item))
                        break

    return frequentSets

def apriori(data,actualSupport,rddLength):
    frequentItemSet = []
    data = list(data)
    #print("Partition Length: "+str(len(data)))
    CSets, baskets  = createBaskets(data)
    #print(CSets, baskets)
    support = math.ceil(actualSupport * len(data) / rddLength)

    temp_count = dict()
    for single in CSets:
        temp_count[single] = 0
        for record in baskets:
            if single in record:
                temp_count[single] += 1
                if temp_count[single]>=support:
                    break
        if temp_count[single]<support:
            del temp_count[single]
    
    LSets = set(sorted(temp_count.keys()))

    sizeOfLset = 1

    while(LSets):
        #print(LSets)
        frequentItemSet.append(LSets)
        
        frequentItems = set()
        if(sizeOfLset==1):
            frequentItems = LSets
        else:
            for i in LSets:
                for j in i:
                    frequentItems.add(j)

        CSets = list()
        for record in baskets:
            commonItems = frequentItems.intersection(record)
            CSets.extend(list(combinations(commonItems,sizeOfLset+1)))
        CSets = set(CSets)

        LSets = createLSets(CSets,baskets,support)
        #print(LSets)
        sizeOfLset +=1
    #print(frequentItemSet)
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
    
    filter_threshold = int(sys.argv[1])
    support = int(sys.argv[2])
    input_file = sys.argv[3]
    output_file = sys.argv[4]
    
    sc= SparkContext('local[*]','Task1')
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

    #print("RDD Length: "+str(rddLength))
    #print(lines.collect())

    #candidates = lines.mapPartitions(lambda x: apriori(x, support, rddLength)).flatMap(lambda x: x).map(lambda x: (x,1)).reduceByKey(add).map(lambda x: x[0]).collect()
    candidates = lines.mapPartitions(lambda x: apriori(x, support, rddLength)).flatMap(lambda x: x).distinct().collect()
    #print(candidates)
    #print(candidates)
    #candidates = list(map(list, candidates))
    formattedCandidates = []
    for i in candidates:
        if type(i)==str:
            formattedCandidates.append([i])
        else:
            formattedCandidates.append(list(i))
    #print(candidates)
    #candidates = lines.mapPartitions(lambda x: apriori(x, support, rddLength)).collect()
    #formattedCandidatesTemp = [i for n, i in enumerate(formattedCandidates) if i not in formattedCandidates[:n]]
    #print(formattedCandidates)
    formattedCandidates = list(formattedCandidates)
    formattedCandidates.sort(key = lambda x: (len(x),x.sort(),x))
    #print(formattedCandidates)
    finalCandidates = sortedPrinting(formattedCandidates)

    frequent = lines.mapPartitions(lambda x: verifyCandidates(x, formattedCandidates)).reduceByKey(add).filter(lambda x: x[1]>=support).map(lambda x: x[0]).collect()
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