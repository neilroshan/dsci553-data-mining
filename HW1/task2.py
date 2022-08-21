import json
from pyspark import SparkContext,SparkConf
from operator import add
import time
import sys

def partitionFunction(key):
    return hash(key)

def task2(review_file_path,noOfPartitions,flag):

    start_time = time.time()
    data = spark.textFile(review_file_path).map(lambda x: json.loads(x))

    if flag==False:
        rdd = data.map(lambda x: (x['business_id'],1))
    elif flag==True:
        rdd = data.map(lambda x: (x['business_id'],1)).partitionBy(int(noOfPartitions),partitionFunction)

    top10Businesses1 = rdd.reduceByKey(add).sortBy(lambda x: (-x[1],x[0])).take(10)
    total_time = time.time() - start_time
    rddNoOfPartitions = rdd.getNumPartitions()
    partitions = rdd.glom().collect()
    listOfItems = [len(x) for x in partitions]

    json_dict = {
        "n_partition": rddNoOfPartitions,
        "n_items": listOfItems,
        "exe_time": total_time
    }

    return json_dict

if __name__=='__main__':
    if len(sys.argv) !=4:
        print("All argumnts haven't been specified")
        sys.exit(-1)
    
    review_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    numPartitions = sys.argv[3]
    
    conf = SparkConf().setAppName("task2").setMaster("local")
    spark = SparkContext(conf=conf)
    spark.setLogLevel("WARN")

    default_rdd = task2(review_file_path,numPartitions,False)
    partitioned_rdd = task2(review_file_path,numPartitions,True)
    output_dict = {"default":default_rdd,"customized":partitioned_rdd}

    with open(output_file_path, "w") as f:
        f.write(json.dumps(output_dict))

    spark.stop()