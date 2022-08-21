from collections import defaultdict
from re import X
from tkinter import NE
from pyspark import SparkConf, SparkContext
import json
import sys
import time
import math
from itertools import combinations
import random
import heapq

NEIGHBOURHOOD = 100
CO_RATED_USER_LIMIT = 9
weights_dict = {}


def to_list(a):
    return [a]

def append(a, b):
    a.append(b)
    return a

def extend(a, b):
    a.extend(b)
    return a

def pearsonSimilarity(pair,ratings):
    try:
        item1_users = set(ratings[pair[0]].keys())
        items2_users = set(ratings[pair[1]].keys())

        co_rated_users = set(item1_users) & set(items2_users)
        if(len(co_rated_users)>CO_RATED_USER_LIMIT):
            item1_ratings = []
            item2_ratings = []  

            for user in co_rated_users:
                item1_ratings.append(float(ratings[pair[0]][user]))
                item2_ratings.append(float(ratings[pair[1]][user]))
            
            item1_average = sum(item1_ratings)/len(item1_ratings)
            item2_average = sum(item2_ratings)/len(item2_ratings)

            item1_rating_average = [i-item1_average for i in item1_ratings]
            item2_rating_average = [i-item2_average for i in item2_ratings]

            numerator = sum([item1_rating_average[i]*item2_rating_average[i] for i in range(len(item1_rating_average))])
            denominator = math.sqrt(sum([i*i for i in item1_rating_average])) * math.sqrt(sum([i*i for i in item2_rating_average]))

            if numerator==0 or denominator==0:
                return 0.5
            return numerator/denominator

        else: 
            return 0.5
    except:
        return 0.5

def getWeight(test_business_id, train_business_id):
    if str(test_business_id)+"_"+str(train_business_id) in weights_dict.keys():
        weight = weights_dict[str(test_business_id)+"_"+str(train_business_id)]
    elif str(train_business_id)+"_"+str(test_business_id) in weights_dict.keys():
        weight = weights_dict[str(train_business_id)+"_"+str(test_business_id)]
    else:
        weight = pearsonSimilarity((test_business_id,train_business_id), ratings)
        weight = weight*pow(abs(weight),2)
    weights_dict[str(train_business_id)+"_"+str(test_business_id)] = weight
    return weight

def sort_function(x):
    x = list(x)
    return sorted(x,key=lambda y: y[1],reverse=True)[0:min(len(x),NEIGHBOURHOOD)]

if __name__=='__main__':
    if len(sys.argv) !=4:
        print("All arguments haven't been specified")
        sys.exit(-1)

    train_file = sys.argv[1]
    test_file = sys.argv[2]
    output_file = sys.argv[3]
    
    sc= SparkContext('local[*]','Task2_1')
    sc.setLogLevel("WARN")

    start_time = time.time()

    train_file = sc.textFile(train_file)
    header = train_file.first()
    
    train_reviews = train_file.filter(lambda x: x!=header).map(lambda x: x.split(",")).map(lambda x: (x[1], x[0], x[2])).distinct().persist()

    businessID_userID_list = train_reviews.map(lambda x: (x[0],x[1])).combineByKey(to_list, append, extend).persist()
    filteredBusinesses = businessID_userID_list.map(lambda x: x[0]).distinct().collect()
    
    ratings = train_reviews.map(lambda x: (x[0], (x[1], x[2]))).combineByKey(to_list, append, extend).map(lambda x: (x[0], dict(list(set(x[1]))))).collectAsMap()

    test_file = sc.textFile(test_file)
    header_test = test_file.first()
    test_businessID_userID_list = test_file.filter(lambda x: x!=header_test).map(lambda x: x.split(",")).map(lambda x: (x[0], x[1])).zipWithIndex().persist()
    # print(test_businessID_userID_list.take(10))

    userID_businessID_dict = train_reviews.map(lambda x: (x[1], x[0])).combineByKey(to_list, append, extend).map(lambda x: (x[0], set(x[1]))).persist().collectAsMap()

    predicted_ratings = test_businessID_userID_list.map(lambda x: ((x[0][0],x[0][1],x[1]),userID_businessID_dict[x[0][0]])).map(lambda x: [(x[0], y) for y in x[1]]).flatMap(lambda x: x).map(lambda x: (x[0], (ratings[x[1]][x[0][0]],getWeight(x[0][1],x[1])))).groupByKey().mapValues(sort_function).map(lambda x: (x[0], sum([w*float(r) for r,w in x[1]])/sum([abs(w) for _,w in x[1]]))).collect()
    
    predicted_ratings = sorted(predicted_ratings,key=lambda x: (x[0][2]))

    with open(output_file, "w") as f:
        f.write("user_id,business_id,prediction\n")
        for i in predicted_ratings:
            f.write(i[0][0]+","+i[0][1]+","+str(i[1]))
            f.write("\n")

    print(time.time()-start_time)
    sc.stop()