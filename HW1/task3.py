import json
from pyspark import SparkContext
from operator import add
import time
import sys

def averageCities(review_file_path,business_file_path,outputA_file_path):
    
    test_review = spark.textFile(review_file_path).map(json.loads).map(lambda x:(x['business_id'], x['stars']))
    business = spark.textFile(business_file_path).map(json.loads).map(lambda x:(x['business_id'], x['city']))
    
    rdd = test_review.join(business)
    data = rdd.map(lambda x: (x[1][1],(x[1][0],1))).reduceByKey(lambda x,y: (x[0]+y[0],x[1]+y[1])).map(lambda x: (x[0],x[1][0]/x[1][1])).sortBy(lambda x: (-x[1],x[0])).collect()
    
    with open(outputA_file_path, "w") as f:
        f.write("city,stars\n")
        for city,stars in data:
            f.write(str(city)+","+str(stars)+"\n")

def top10Cities(review_file_path,business_file_path,flag):
    
    start_time = time.time()

    test_review = spark.textFile(review_file_path).map(json.loads).map(lambda x:(x['business_id'], x['stars']))
    business = spark.textFile(business_file_path).map(json.loads).map(lambda x:(x['business_id'], x['city']))
    rdd = test_review.join(business)
    
    if(flag=="Spark"):
        data = rdd.map(lambda x: (x[1][1],(x[1][0],1))).reduceByKey(lambda x,y: (x[0]+y[0],x[1]+y[1])).map(lambda x: (x[0],x[1][0]/x[1][1])).sortBy(lambda x: (-x[1],x[0])).take(10)
    elif(flag=="Python"):
        data = rdd.map(lambda x: (x[1][1],(x[1][0],1))).reduceByKey(lambda x,y: (x[0]+y[0],x[1]+y[1])).map(lambda x: (x[0],x[1][0]/x[1][1])).collect()
        sorted_data = sorted(data,key=lambda x: (-x[1], x[0]))[:10]
    
    return time.time() - start_time

def helper(review_file_path,business_file_path,outputB_file_path):
    python_time = top10Cities(review_file_path,business_file_path,"Python")
    spark_time = top10Cities(review_file_path,business_file_path,"Spark")
    output_dict = {
                    "m1":python_time,
                    "m2":spark_time, 
                    "reason": "abc"
                }
    json_object = json.dumps(output_dict)
    
    with open(outputB_file_path, "w") as f:
        f.write(json_object)


if __name__=='__main__':
    if len(sys.argv) !=5:
        print("All arguments haven't been specified")
        sys.exit(-1)
    
    review_file_path = sys.argv[1]
    business_file_path = sys.argv[2]
    outputA_file_path = sys.argv[3]
    outputB_file_path = sys.argv[4]
    
    spark = SparkContext('local[*]', 'task3')
    spark.setLogLevel("WARN")
    
    averageCities(review_file_path,business_file_path,outputA_file_path)
    helper(review_file_path,business_file_path,outputB_file_path)
    
    spark.stop()




