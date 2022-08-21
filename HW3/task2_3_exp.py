from collections import defaultdict
from pyspark import SparkConf, SparkContext
import json
import sys
import time
import math
from itertools import combinations
import random
import xgboost as xg
import pandas as pd
from collections import defaultdict
import time

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

def getWeight(test_business_id, train_business_id, ratings):
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


def itemBased():

    folder_path = sys.argv[1]
    test_file = sys.argv[2]
    output_file = "item_based.csv"

    train_file = sc.textFile(folder_path+'/yelp_train.csv')
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

    predicted_ratings = test_businessID_userID_list.map(lambda x: ((x[0][0],x[0][1],x[1]),userID_businessID_dict[x[0][0]])).map(lambda x: [(x[0], y) for y in x[1]]).flatMap(lambda x: x).map(lambda x: (x[0], (ratings[x[1]][x[0][0]],getWeight(x[0][1],x[1],ratings)))).groupByKey().mapValues(sort_function).map(lambda x: (x[0], sum([w*float(r) for r,w in x[1]])/sum([abs(w) for _,w in x[1]]))).collect()
    
    predicted_ratings = sorted(predicted_ratings,key=lambda x: (x[0][2]))

    with open(output_file, "w") as f:
        f.write("user_id,business_id,prediction\n")
        for i in predicted_ratings:
            f.write(i[0][0]+","+i[0][1]+","+str(i[1]))
            f.write("\n")


def getDataset(folderPath,chunkSize,case):
    chunkIterator = pd.read_json(folderPath, lines=True, chunksize=chunkSize)
    chunk_list = []
    for chunk in chunkIterator:
        if case=='user':
            chunk_filter = chunk[['user_id','average_stars','review_count','useful','fans']]
        else:
            chunk_filter = chunk[['business_id','stars','review_count']]
        chunk_list.append(chunk_filter)
    return pd.concat(chunk_list)


def modelBased():
    folder_path = sys.argv[1]
    test_file = sys.argv[2]
    output_file = "model_based.csv"

    user_df = getDataset(folder_path+"/user.json",100000,'user')
    busniness_df = getDataset(folder_path+"/business.json",100000,'business')

    user_df = user_df.rename({'review_count':'user_review_count'}, axis=1)
    busniness_df = busniness_df.rename({'stars':'business_stars','review_count':'business_review_count'}, axis=1)

    train_df = pd.read_csv(folder_path+'/yelp_train.csv')
    train_df = pd.merge(train_df,user_df,on='user_id',how='inner')
    train_df = pd.merge(train_df,busniness_df,on='business_id',how='inner')

    user_rating_info = defaultdict(dict)
    for index, row in train_df.iterrows():
        if row['user_id'] not in user_rating_info.keys():
            user_rating_info[row['user_id']] = {}
            user_rating_info[row['user_id']]['std_sum'] = pow(row['stars'] - row['average_stars'],2)
            user_rating_info[row['user_id']]['min'] = row['stars']
            user_rating_info[row['user_id']]['max'] = row['stars']
            user_rating_info[row['user_id']]['no_of_reviews'] = 1
        else:
            user_rating_info[row['user_id']]['std_sum'] = user_rating_info[row['user_id']]['std_sum']+pow(row['stars'] - row['average_stars'],2)
            user_rating_info[row['user_id']]['min'] = row['stars'] if row['stars']<user_rating_info[row['user_id']]['min'] else user_rating_info[row['user_id']]['min']
            user_rating_info[row['user_id']]['max']= row['stars'] if row['stars']>user_rating_info[row['user_id']]['max'] else user_rating_info[row['user_id']]['max']
            user_rating_info[row['user_id']]['no_of_reviews'] += 1

    for key,val in user_rating_info.items():
        user_rating_info[key]['std'] = user_rating_info[key]['std_sum'] / user_rating_info[key]['no_of_reviews']

    user_rating_df = pd.DataFrame.from_dict(user_rating_info,orient='index')
    user_rating_df.index.name = 'user_id'
    train_df = pd.merge(train_df,user_rating_df,on='user_id', how='inner')
    train_df = train_df.drop(['std_sum', 'no_of_reviews'], axis=1)

    train_df_features = train_df[['average_stars', 'user_review_count','useful', 'fans', 'business_stars', 'business_review_count', 'min','max', 'std']]
    train_df_target = train_df[['stars']]

    model = xg.XGBRegressor()
    model.fit(train_df_features,train_df_target)

    test_df = pd.read_csv(test_file)
    test_df = pd.merge(test_df,user_df,on='user_id',how='left')
    test_df = pd.merge(test_df,busniness_df,on='business_id',how='left')
    test_df = pd.merge(test_df,user_rating_df,on='user_id', how='left')
    test_df = test_df.drop(['std_sum', 'no_of_reviews'], axis=1)

    test_users_id = test_df['user_id']
    test_business_id = test_df['business_id']
    #test_stars = test_df['stars']
    test_df = test_df.drop(['user_id','business_id','stars'],axis=1)
    #print(test_df.columns)
    
    rating_prediction = model.predict(test_df)

    predicted_df = pd.DataFrame()
    predicted_df['user_id'] = test_users_id
    predicted_df['business_id'] = test_business_id
    predicted_df['prediction'] = rating_prediction

    with open(output_file, "w") as f:
        f.write("user_id,business_id,prediction\n")
        for index,row in predicted_df.iterrows():
            f.write(row['user_id']+','+row['business_id']+','+str(row['prediction']))
            f.write("\n")

if __name__=='__main__':
    if len(sys.argv) !=4:
        print("All arguments haven't been specified")
        sys.exit(-1)
    ALPHA = 0.01
    folder_path = sys.argv[1]
    test_file = sys.argv[2]
    output_file = sys.argv[3]

    start_time = time.time()

    sc= SparkContext('local[*]','Task2_3')
    sc.setLogLevel("WARN")

    train_dfCopy = pd.read_csv(folder_path+'/yelp_train.csv')
    train_dfCopy = train_dfCopy.groupby(['user_id']).size().reset_index(name='counts')
    maxNoOfBusinessesRated = train_dfCopy['counts'].max()
    train_dfCopy['alphas'] = train_dfCopy['counts']/maxNoOfBusinessesRated
    train_dfCopy = train_dfCopy.drop(['counts'],axis=1)

    userID_Alphas = train_dfCopy.set_index('user_id').T.to_dict()

    itemBased()
    modelBased()

    item_based_ratings = open('item_based.csv','r')
    model_based_ratings = open('model_based.csv','r')

    with open(output_file,'w') as f:
        f.write("user_id,business_id,prediction\n")
        
        item_based_rating = item_based_ratings.readline()
        model_based_rating = model_based_ratings.readline()
        
        while(True):
            item_based_rating = item_based_ratings.readline()
            model_based_rating = model_based_ratings.readline()

            if not item_based_rating or not model_based_rating: 
                break
            
            user_id, business_id, item_rating = item_based_rating.split(',')    
            _, _, model_rating = model_based_rating.split(',')

            if(user_id in userID_Alphas.keys()):
                alpha = userID_Alphas[user_id]['alphas']
            else:
                alpha = 0.01

            finalRating = (alpha*float(item_rating)) + ((1-alpha)*float(model_rating)) 
            f.write(user_id+","+business_id+","+str(finalRating))
            f.write("\n")

        item_based_ratings.close()
        model_based_ratings.close()
    
    print(time.time()-start_time)