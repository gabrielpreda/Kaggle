import os
import tweepy as tw
import pandas as pd
from tqdm import tqdm


def twitter_connection():
    consumer_api_key = os.environ["TWITTER_CONSUMER_API_KEY"]
    consumer_api_secret = os.environ["TWITTER_CONSUMER_API_SECRET"]

    auth = tw.OAuthHandler(consumer_api_key, consumer_api_secret)
    api = tw.API(auth, wait_on_rate_limit=True)
    return api


def create_cursor(api, search_words, date_since, language="en", items_limit=3000):
    
    # Collect tweets
    tweets = tw.Cursor(api.search,
                  q=search_words,
                  lang=language,
                  since=date_since).items(items_limit)


    print(f"retreive new tweets ...")
    tweets_copy = []
    for tweet in tqdm(tweets):
        tweets_copy.append(tweet)
     
    print(f"new tweets retrieved: {len(tweets_copy)}")

    return tweets_copy
   

def build_dataset(tweets_copy):
    tweets_df = pd.DataFrame()
    for tweet in tqdm(tweets_copy):
        hashtags = []
        try:
            for hashtag in tweet.entities["hashtags"]:
                hashtags.append(hashtag["text"])
        except:
            pass
        tweets_df = tweets_df.append(pd.DataFrame({'id': tweet.id,
                                                   'user_name': tweet.user.name, 
                                                   'user_location': tweet.user.location,\
                                                   'user_description': tweet.user.description,
                                                   'user_created': tweet.user.created_at,
                                                   'user_followers': tweet.user.followers_count,
                                                   'user_friends': tweet.user.friends_count,
                                                   'user_favourites': tweet.user.favourites_count,
                                                   'user_verified': tweet.user.verified,
                                                   'date': tweet.created_at,
                                                   'text': tweet.text, 
                                                   'hashtags': [hashtags if hashtags else None],
                                                   'source': tweet.source,
                                                   'retweets': tweet.retweet_count,
                                                   'favorites': tweet.favorite_count,
                                                   'is_retweet': tweet.retweeted}, index=[0]))
    return tweets_df


def update_and_save_dataset(tweets_df):   
    file_path = "vaccination_tweets.csv"
    if os.path.exists(file_path):
        tweets_old_df = pd.read_csv(file_path)
        print(f"past tweets: {tweets_old_df.shape}")
        tweets_all_df = pd.concat([tweets_old_df, tweets_df], axis=0)
        print(f"new tweets: {tweets_df.shape[0]} past tweets: {tweets_old_df.shape[0]} all tweets: {tweets_all_df.shape[0]}")
        tweets_new_df = tweets_all_df.drop_duplicates(subset = ["id"], keep='last', inplace=False)
        print(f"all tweets: {tweets_new_df.shape}")
        tweets_new_df.to_csv(file_path, index=False)
    else:
        print(f"tweets: {tweets_df.shape}")
        tweets_df.to_csv(file_path, index=False)
    
if __name__ == "__main__":        
    api = twitter_connection()
    tweets_copy = create_cursor(api, "#PfizerBioNTech -filter:retweets", "2020-10-01")
    tweets_df = build_dataset(tweets_copy)
    update_and_save_dataset(tweets_df)
