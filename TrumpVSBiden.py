#!/usr/bin/env python
# coding: utf-8

# In[30]:


import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast


# # Data

# In[31]:


tweets_data_path = 'twitter_data.txt'

tweets_data = []
tweets_file = open(tweets_data_path, "r")
for line in tweets_file:
    try:
        tweet = json.loads(line)
        tweets_data.append(tweet)
    except:
        continue


# In[32]:


len(tweets_data)


# In[33]:


tweets_data


# In[34]:


tweets = pd.DataFrame(tweets_data,columns=['text','lang','extended_tweet','timestamp_ms'])
tweets.head()


# In[35]:


tweets["extended_tweet"] = tweets["extended_tweet"].apply(lambda x : ast.literal_eval(x) if type(x) is str else np.nan)


# In[36]:


tweets["complete_tweets"] = tweets.apply(lambda x: x['text'] if pd.isnull(x['extended_tweet']) else x['extended_tweet']["full_text"], axis=1)


# In[37]:


tweets.head()


# In[38]:


tweets.drop(["text", "lang", "extended_tweet"], 1, inplace=True)


# In[39]:


tweets.head()


# In[40]:


tweets.dropna(inplace=True)


# # Preprocessing

# In[41]:


import nltk
from nltk.corpus import stopwords
from textblob import Word, TextBlob


# In[42]:


nltk.download("stopwords")
nltk.download("wordnet")
stop_words = stopwords.words("english")
custom_stopwords = ["RT"]


# In[43]:


def preprocess_tweets(tweet, custom_stopwords):
    preprocessed_tweet = tweet
    preprocessed_tweet.replace('[^\w\s]', '')
    preprocessed_tweet = " ".join(Word(word).lemmatize() for word in preprocessed_tweet.split() if  word not in custom_stopwords)
    return preprocessed_tweet


# In[44]:


tweets["processed_tweet"] = tweets["complete_tweets"].apply(lambda tweet : preprocess_tweets(tweet, custom_stopwords))


# In[45]:


tweets.head()


# In[46]:


trump_refs = ["DonaldTrump", "Donald Trump", "Donald", "Trump", "trump"]
biden_refs = ["JoeBiden", "Joe Biden", "Joe", "Biden", "biden"]


# In[47]:


def identify_subject(tweet, refs):
    flag = 0
    for ref in refs:
        if tweet.find(ref) != -1:
            flag = 1
    return flag


# In[48]:


tweets["trump"] = tweets["complete_tweets"].apply(lambda tweet : identify_subject(tweet, trump_refs))


# In[49]:


tweets["biden"] = tweets["complete_tweets"].apply(lambda tweet : identify_subject(tweet, biden_refs))


# In[50]:


tweets.dropna(inplace=True)


# In[51]:


tweets.shape


# # Sentiment analysis

# In[52]:


tweets["polarity"] = tweets["processed_tweet"].apply(lambda tweet : TextBlob(tweet).sentiment[0] if tweet != np.nan else np.nan)
tweets["subjectivity"] = tweets["processed_tweet"].apply(lambda tweet : TextBlob(tweet).sentiment[1] if tweet != np.nan else np.nan)


# In[53]:


tweets.head()


# # Visualize

# In[54]:


biden = tweets[tweets["biden"] == 1.0][["timestamp_ms", "polarity"]]
biden = biden.sort_values(by="timestamp_ms", ascending=True)
biden["mean_polarity"] = biden.polarity.rolling(100, min_periods = 3).mean()


# In[55]:


trump = tweets[tweets["trump"] == 1.0][["timestamp_ms", "polarity"]]
trump = trump.sort_values(by="timestamp_ms", ascending=True)
trump["mean_polarity"] = trump.polarity.rolling(100, min_periods = 3).mean()


# In[56]:


fig, axes = plt.subplots(2, 1, figsize=(15, 10))
axes[0].plot(biden["timestamp_ms"][20:], biden["mean_polarity"][20:], "b")
axes[0].set_title("\n".join(["Biden 10 Tweet Moving Average Polarity"]))
axes[1].plot(trump["timestamp_ms"][20:], trump["mean_polarity"][20:], "r")
axes[1].set_title("\n".join(["Trump 10 Tweet Moving Average Polarity"]))
fig.suptitle("\n".join(["Presidential Analysis"]), y=0.98)


# In[ ]:




