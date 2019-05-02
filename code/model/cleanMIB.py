import pandas as pd
import numpy as np
import io
import requests
import math
import re

#claen data
#status: -1 represent private account, 1 represent public account
def input_data(status, user_true_path, user_false_path, tweets_true_path, tweets_false_path, follower_threshold, friend_threshold):

    #user preprocessing
    user_cols = ["id", "screen_name", "followers_count", "friends_count", "location", "url", "profile_banner_url", "default_profile", "default_profile_image"]
    user_true = pd.read_csv(user_true_path, usecols = user_cols)
    user_false = pd.read_csv(user_false_path, usecols = user_cols)

    #combine user
    label = []
    for i in range(0, user_true.shape[0]):
        label.append(1)
    user_true['label'] = label
    #print(user_true.head(3))
    label = []
    for i in range(0, user_false.shape[0]):
        label.append(0)
    user_false['label'] = label
    #print(user_false.head(3))
    frames = [user_false, user_true]
    user = pd.concat(frames)
    #print(user.head(5))
    user = user.fillna(" ")


    #has_banner
    #print(user['profile_banner_url'].head(5))
    user['has_banner'] = user['profile_banner_url'].apply(lambda x: 1 if x != " " else 0)
    #print(user['has_banner'].head(5))
    #has_location
    #print(user['location'].head(5))
    user['has_location'] = user['location'].apply(lambda x: 1 if x != " " else 0)
    #print(user['has_location'].head(5))
    #has_extended_profile
    #print(user['default_profile'].head(5))
    user['has_extended_profile'] = user['default_profile'].apply(lambda x: 1 if x == " " else 0)
    #print(user['has_extended_profile'].head(5))
    #has_url
    #print(user['url'].head(5))
    user['has_url'] = user['url'].apply(lambda x: 1 if x != " " else 0)
    #print(user['has_url'].head(5))
    #user_name, user_id
    user = user.rename(columns = {'screen_name':'user_name'})
    user = user.rename(columns = {'id':'user_id'})
    user = user.rename(columns = {'followers_count':'followers'})
    user = user.rename(columns = {'friends_count':'friends'})
    #followers
    #print(user['followers_count'].head(5))
    #user['followers'] = user['followers_count'].apply(lambda x: 1 if x > follower_threshold else 0)
    #print(user['followers'].head(5))
    #friends
    #print(user['friends_count'].head(5))
    #user['friends'] = user['friends_count'].apply(lambda x: 1 if x > friend_threshold else 0)
    #print(user['friends'].head(5))
    print(user.shape)
    #has_number

    #has_number_at_end

    #alpha_numeric_ratio

    #friends_followers_ratio

    if status == -1:
        return user[['has_banner', 'has_location', 'has_extended_profile', 'has_url', 'followers', 'friends', 'label']].sample(frac=1)

    #tweets preprocessing
    tweet_cols = ["user_id", "text", "retweet_count", "num_urls", "num_hashtags"]
    tweets_true = pd.read_csv(tweets_true_path, usecols = tweet_cols)
    tweets_false = pd.read_csv(tweets_false_path, usecols = tweet_cols)
    #print(tweets_true.head(5))

    #combine tweets
    frames = [tweets_false, tweets_true]
    tweets = pd.concat(frames)
    tweets['text'] = tweets['text'].fillna(" ")
    #print(tweets.head(5))
    #has_retweet
    #print(tweets['retweet_count'].head(25))
    tweets['has_retweet'] = tweets['retweet_count'].apply(lambda x: 1 if x > 0 else 0)
    #print(tweets['has_retweet'].head(25))
    #contains_hashTag
    #print(tweets['num_hashtags'].head(25))
    tweets['contains_hashTag'] = tweets['num_hashtags'].apply(lambda x: 1 if x > 0 else 0)
    #print(tweets['contains_hashTag'].head(25))
    #has_Duplicate, onlyHasUrl
    print(tweets.shape)

    def group_by_user(data):
        #print(data.shape)
        onlyHasUrl = 0
        has_Duplicate = 0
        has_retweet = 0
        contains_hashTag = 0
        num_posts = 0
        pre = " "
        for index, row in data.iterrows():
            text = row['text']
            #print(row['text'])
            rm_tmp = re.sub(r"http\S+", "", text)
            rm_tmp = re.sub(r"https\S+", "", rm_tmp)
            #print(rm_tmp)
            if rm_tmp == " " or rm_tmp == "":
                onlyHasUrl += 1
            if rm_tmp == pre:
                has_Duplicate += 1
            pre = rm_tmp
            if row['has_retweet'] == 1:
                has_retweet += 1
            if row['contains_hashTag'] == 1:
                contains_hashTag += 1
            num_posts += 1
        data['onlyHasUrl'] = onlyHasUrl
        data['has_Duplicate'] = has_Duplicate
        data['has_retweet'] = has_retweet
        data['contains_hashTag'] = contains_hashTag
        data['num_posts'] = num_posts
        data['user_id'] = int(data['user_id'].head(1))

        #print(onlyHasUrl)
        #print(has_Duplicate)
        return data[['user_id', 'num_posts', 'has_Duplicate', 'onlyHasUrl', 'has_retweet', 'contains_hashTag']].head(1)



    tweets = tweets.groupby('user_id', sort=False, as_index=False).apply(group_by_user)
    tweets = tweets[['user_id', 'num_posts', 'has_Duplicate', 'onlyHasUrl', 'has_retweet', 'contains_hashTag']]
    print(tweets.shape)
    #concatanate user and tweets
    #res = pd.concat([user, tweets], axis=1, ignore_index=True, sort=False, join='inner')
    res = user.merge(tweets, on='user_id')
    res = res[['has_banner', 'has_location', 'has_extended_profile', 'has_url', 'followers', 'friends', 'num_posts', 'has_Duplicate', 'onlyHasUrl', 'has_retweet', 'contains_hashTag', 'label']]
    print(res.shape)
    #print(res.head(5))
    #print(res.tail(5))
    return res.tail(2200)

user_true_path = "datasets_full/datasets_full.csv/genuine_accounts.csv/users.csv"
tweets_true_path = "datasets_full/datasets_full.csv/genuine_accounts.csv/tweets.csv"
user_false_path = "datasets_full/datasets_full.csv/fake_followers.csv/users.csv"
tweets_false_path = "datasets_full/datasets_full.csv/fake_followers.csv/tweets.csv"
follower_threshold = 30
friend_threshold = 100
#dataset = input_data(-1, user_true_path, user_false_path, tweets_true_path, tweets_false_path, follower_threshold, friend_threshold)
#dataset.to_csv("clean/user_all.csv", index=False)
dataset = input_data(1, user_true_path, user_false_path, tweets_true_path, tweets_false_path, follower_threshold, friend_threshold)
dataset.to_csv("clean/user_public.csv", index=False)
