import sys
import time
from graphframes import GraphFrame
import os
import itertools
from pyspark import SparkContext
from pyspark.sql import SparkSession

#os.environ["PYSPARK_SUBMIT_ARGS"] = ("--packages graphframes:graphframes:0.8.2-spark3.1-s_2.12")

def to_list(a):
    return [a]

def append(a, b):
    a.append(b)
    return a

def extend(a, b):
    a.extend(b)
    return a

def checkThreshold(user1, user2, threshold):
    if len(user1&user2) >= threshold:
        return True
    else:
        return False

if __name__=='__main__':
    if len(sys.argv) !=4:
        print("All arguments haven't been specified")
        sys.exit(-1)

    threshold = int(sys.argv[1])
    input_file = sys.argv[2]
    output_file = sys.argv[3]
    
    sc= SparkContext('local[*]','Task1')
    sc.setLogLevel("WARN")

    start_time = time.time()

    train_file = sc.textFile(input_file)
    header = train_file.first()

    userID_businessID_dict = train_file.filter(lambda x: x!=header).map(lambda x: x.split(",")).map(lambda x: (x[0], x[1])).combineByKey(to_list, append, extend).map(lambda x: (x[0], set(x[1]))).persist().collectAsMap()

    businessID_userID_dict = train_file.filter(lambda x: x!=header).map(lambda x: x.split(",")).map(lambda x: (x[1], x[0])).combineByKey(to_list, append, extend).map(lambda x: (x[0], set(x[1]))).persist().collectAsMap()

    userIDPairs = set(list(itertools.combinations(list(userID_businessID_dict.keys()), 2)))
    vertices = set()
    edges = set()

    for userPair in userIDPairs:
        if(checkThreshold(set(userID_businessID_dict[userPair[0]]),set(userID_businessID_dict[userPair[1]]),threshold)):
            edges.add((userPair[0],userPair[1]))
            edges.add((userPair[1],userPair[0]))
            vertices.add(userPair[0])
            vertices.add(userPair[1])
    
    verticesRDD = sc.parallelize(list(vertices))
    edgesRDD = sc.parallelize(list(edges))

    spark = SparkSession(sc)

    verticesDF = verticesRDD.map(lambda x: (x,)).toDF(['id'])
    edgesDF = edgesRDD.toDF(['src','dst'])
    
    g = GraphFrame(verticesDF, edgesDF)
    #print(g)

    result = g.labelPropagation(maxIter=5).rdd.map(lambda x: (x[1],x[0])).combineByKey(to_list, append, extend).map(lambda x: sorted(x[1])).sortBy(lambda x: (len(x),x[0])).collect()

    #print(result)
    with open(output_file, "w") as f:
        for community in result:
            output_str = str(community).strip('[]')
            f.write(output_str)
            f.write('\n')
    
    print(time.time()-start_time)