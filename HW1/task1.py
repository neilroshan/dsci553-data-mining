import json
from pyspark import SparkContext
from operator import add
import sys

def task1(review_file_path,output_file_path):
    lines = spark.textFile(review_file_path).map(json.loads)

    total_noOfReviews = lines.count()
    
    reviews2018 = lines.filter(lambda x: '2018' in x['date'].split('-')[0]).map(lambda x: (1,1)).reduceByKey(add).collect()[0][1]
    
    noOfDistinctUsers = lines.map(lambda x: (x['user_id'],1)).distinct().count()
    
    noOfDistinctBusiness = lines.map(lambda x: (x['business_id'],1)).distinct().count()

    top10UsersList = []
    top10Users = lines.map(lambda x: (x['user_id'],1)).reduceByKey(add).sortBy(lambda x: (-x[1],x[0])).take(10)
    for users,ratings in top10Users:
        top10UsersList.append([users,ratings])

    top10Businesses = lines.map(lambda x: (x['business_id'],1)).reduceByKey(add).sortBy(lambda x: (-x[1],x[0])).take(10)
    top10BusinessesList = []
    for business,ratings in top10Businesses:
        top10BusinessesList.append([business,ratings])
    
    output_dict = {
                    "n_review":total_noOfReviews,
                    "n_review_2018":reviews2018,
                    "n_user":noOfDistinctUsers,
                    "top10_user":top10UsersList,
                    "n_business":noOfDistinctBusiness,
                    "top10_business":top10BusinessesList
    }

    json_object = json.dumps(output_dict)
    
    with open(output_file_path, "w") as f:
        f.write(json_object)

if __name__=='__main__':
    if len(sys.argv) !=3:
        print("All arguments haven't been specified")
        sys.exit(-1)
    
    review_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    
    spark = SparkContext('local[*]', 'task1')
    spark.setLogLevel("WARN")
    
    task1(review_file_path,output_file_path)

    spark.stop()




