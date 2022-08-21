import xgboost as xg
import numpy as np
import pandas as pd
import sys
from collections import defaultdict
import time

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

if __name__=='__main__':
    if len(sys.argv) !=4:
        print("All arguments haven't been specified")
        sys.exit(-1)

    folder_path = sys.argv[1]
    test_file = sys.argv[2]
    output_file = sys.argv[3]

    start_time = time.time()

    
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

    test_df = pd.read_csv(folder_path+'/yelp_val.csv')
    test_df = pd.merge(test_df,user_df,on='user_id',how='left')
    test_df = pd.merge(test_df,busniness_df,on='business_id',how='left')
    test_df = pd.merge(test_df,user_rating_df,on='user_id', how='left')
    test_df = test_df.drop(['std_sum', 'no_of_reviews'], axis=1)

    test_users_id = test_df['user_id']
    test_business_id = test_df['business_id']
    test_stars = test_df['stars']
    test_df = test_df.drop(['user_id','business_id','stars'],axis=1)

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

    print(time.time()-start_time)