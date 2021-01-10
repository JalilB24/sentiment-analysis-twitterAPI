#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import time
import ast


# In[2]:

import re
from textblob import Word, TextBlob


# In[3]:


import tweepy as tw
from tweepy.streaming import StreamListener
from tweepy import Stream


# In[4]:


access_token = "ENTER YOUR ACCESS TOKEN"
access_token_secret = "ENTER YOUR SECRET ACCESS TOKEN"
consumer_key = "ENTER YOUR CONSUMER KEY"
consumer_secret = "ENTER YOUR SECRET CONSUMER KEY"


# In[5]:


auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)


# In[6]:


def preproccess_dataframe(dataframe):
    
    tweets = dataframe.copy()
    tweets = tweets[tweets["lang"] == "en"]
    tweets["extended_tweet"] = tweets["extended_tweet"].apply(lambda x : ast.literal_eval(x) if type(x) is str else np.nan)
    tweets["complete_tweets"] = tweets.apply(lambda x: x['text'] if pd.isnull(x['extended_tweet']) else x['extended_tweet']["full_text"], axis=1)
    tweets.drop(["text", "lang", "extended_tweet"], axis=1, inplace=True)
    tweets.reset_index(inplace=True)
    tweets["processed_tweet"] = tweets["complete_tweets"].apply(lambda tweet : preprocess_tweets(tweet, custom_stopwords=["RT"]))
    tweets["polarity"] = tweets["processed_tweet"].apply(lambda tweet : TextBlob(tweet).sentiment[0] if tweet != np.nan else np.nan)
    tweets["subjectivity"] = tweets["processed_tweet"].apply(lambda tweet : TextBlob(tweet).sentiment[1] if tweet != np.nan else np.nan)
    tweets["trump"] = tweets["processed_tweet"].apply(lambda tweet : identify_subject(tweet, ["donaldtrump", "donald trump", "donald", "trump"]))
    tweets["biden"] = tweets["processed_tweet"].apply(lambda tweet : identify_subject(tweet, ["joebiden", "joe biden", "joe", "biden"]))
    tweets["timestamp_ms"] = pd.to_datetime(tweets["timestamp_ms"], unit='ms')
    
    return tweets


# In[7]:


def preprocess_tweets(tweet, custom_stopwords=["RT"]):
    
    preprocessed_tweet = tweet
    preprocessed_tweet = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', preprocessed_tweet)
    preprocessed_tweet = re.sub('[^\w\s@]', '', preprocessed_tweet)
    preprocessed_tweet = " ".join(Word(word).lemmatize() for word in preprocessed_tweet.split() if  word not in custom_stopwords)
    
    return preprocessed_tweet.lower()


# In[8]:


def process_tweets(dataframe):
    
    tweets = dataframe.copy()
    tweets["processed_tweet"] = tweets["tweet"].apply(lambda tweet : preprocess_tweets(tweet, custom_stopwords=["RT"]))
    tweets["polarity"] = tweets["processed_tweet"].apply(lambda tweet : TextBlob(tweet).sentiment[0] if tweet != np.nan else np.nan)
    tweets["subjectivity"] = tweets["processed_tweet"].apply(lambda tweet : TextBlob(tweet).sentiment[1] if tweet != np.nan else np.nan)
    
    return tweets


# In[9]:


def get_tweets(hashtag, num_tweets=400):
    query = tw.Cursor(api.search, q=hashtag, tweet_mode='extended').items(num_tweets)
    tweets = []
    for tweet in query:
        tweets.append({"tweet" : tweet.full_text, "time_stamp": tweet.created_at})
    return pd.DataFrame.from_dict(tweets)


# In[10]:


def identify_subject(tweet, refs):
    flag = 0
    for ref in refs:
        if tweet.find(ref) != -1:
            flag = 1
    return flag


# In[11]:


tweets = pd.read_csv("tweets.csv")


# In[12]:


tweets = preproccess_dataframe(tweets)


# In[13]:


def predict_sentiment(tweet):
    polarity = TextBlob(preprocess_tweets(tweet, custom_stopwords)).sentiment[0]
    if polarity == 0.0:
        return (0, polarity, "neutral")
    elif polarity < 0.0:
        return (-1, polarity, "negative")
    else:
        return (1, polarity, "positive")


# # Visualize

# In[14]:


import plotly.graph_objs as go
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc


# In[15]:


def plot_polarity_subjectivity(dataframe, keyword="polarity"):
    if keyword == "both":
        plot_data = []
        dataframe[f"mean_polarity"] = dataframe["polarity"].rolling(dataframe.shape[0] // 100, min_periods=1).mean()
        dataframe[f"mean_subjectivity"] = dataframe["subjectivity"].rolling(dataframe.shape[0] // 100, min_periods=1).mean()
        plot_data.append(go.Scatter(x=np.arange(0, dataframe.shape[0])[10:], y=dataframe[f"mean_polarity"][10:],
                               mode="lines",
                               name="polarity",
                               line=dict(color="firebrick")))
        plot_data.append(go.Scatter(x=np.arange(0, dataframe.shape[0])[10:], y=dataframe[f"mean_subjectivity"][10:],
                               mode="lines",
                               name="subjectivity",
                               line=dict(color="royalblue")))
        keyword = "Polarity and Subjectivity"
        layout = go.Layout(title=f"Average {keyword}",
                      xaxis=dict(title="time"),
                      yaxis=dict(title=keyword))
        fig = go.Figure(data=plot_data, layout=layout)
    else:
        color_choice = np.random.choice(["firebrick", "royalblue"], size=1)[0]
        dataframe[f"mean_{keyword}"] = dataframe[keyword].rolling(dataframe.shape[0] // 100, min_periods=1).mean()
        plot_data = []
        plot_data.append(go.Scatter(x=np.arange(0, dataframe.shape[0])[10:], y=dataframe[f"mean_{keyword}"][10:],
                                   mode="lines",
                                   name=keyword,
                                   line=dict(color=color_choice)))
        layout = go.Layout(title=f"Average {keyword}",
                          xaxis=dict(title="time"),
                          yaxis=dict(title=keyword))



        fig = go.Figure(data=plot_data, layout=layout)
    return fig


# In[16]:


def plot_trump_biden(keyword="polarity"):
    
    trump = tweets[tweets["trump"] == 1.0][["timestamp_ms", keyword]]
    trump = trump.sort_values(by="timestamp_ms", ascending=True)
    trump[f"mean_{keyword}"] = trump[keyword].rolling(1000, min_periods = 3).mean()
    
    biden = tweets[tweets["biden"] == 1.0][["timestamp_ms", keyword]]
    biden = biden.sort_values(by="timestamp_ms", ascending=True)
    biden[f"mean_{keyword}"] = biden[keyword].rolling(1000, min_periods = 3).mean()
    
    plot_data = []
    plot_data.append(go.Scatter(x=trump["timestamp_ms"][100:], y=trump[f"mean_{keyword}"][100:],
                               mode="lines",
                               name=f"Trump {keyword}",
                               line=dict(color="firebrick")))
    
    
    plot_data.append(go.Scatter(x=biden["timestamp_ms"][100:], y=biden[f"mean_{keyword}"][100:],
                               mode="lines",
                               name=f"Biden {keyword}",
                               line=dict(color="royalblue")))
    
    layout = go.Layout(title=f"10 Tweet Moving Average {keyword}",
                      xaxis=dict(title="Time"),
                      yaxis=dict(title=keyword))

    fig = go.Figure(data=plot_data, layout=layout)
    return fig


# In[17]:


init_fig_trump_biden = plot_trump_biden("subjectivity")


# In[18]:


dataframe_holder = None

app = dash.Dash()
server=app.server

app.layout = html.Div([
    html.H1("Twitter Tracking System", style={"text-align":"center"}),
    html.Br(),
    
    html.Div([
            html.Br(),
            html.H2("Trump VS Biden", style={"text-align":"center"}),
            html.Br(),
            dcc.Dropdown(id="pola_sub",
                        options=[
                            dict(label="Polarity", value="polarity"),
                            dict(label="Subjectivity", value="subjectivity")],
                            multi=False,
                            value="polarity",
                            style={"width":"40%"}
                        ),

            dcc.Graph(id="trump_biden", figure=init_fig_trump_biden)
        ]),
    
    html.Br(),
    
    html.Div([
        html.Br(),
        html.H2("Search for a specific subject in this section", style={"text-align":"center"}),
        html.Br(),
        html.Div(dbc.Input(id="search_keyword", 
                            type="text", placeholder="Search Keyword", debounce=True, style=dict(width="80%", height="25px", margin="2px 0px 10px 0px")), style=dict(width="60%", display="inline-block")),
        html.Div(dcc.Dropdown(id="search_choice",
                        options=[
                            dict(label="Polarity", value="polarity"),
                            dict(label="Subjectivity", value="subjectivity"),
                            dict(label="Both", value="both")],
                            multi=False,
                            value="polarity",
                        ), style=dict(width="40%", display="inline-block")),
        html.Br(),
        html.Div(id="search_output")
    ], style=dict(margin="0px 0px 100px 0px"))
])

@app.callback(
    Output("trump_biden", "figure"),
    Input("pola_sub", "value"),
)
def update_bar(value):
    return plot_trump_biden(keyword = value)

@app.callback(
    Output("search_output", "children"),
    Input("search_keyword", "value"),
)

def update_search_output(value):
    print(value)
    if value :
        tweets = get_tweets(value)
        print("generated tweets")
        tweets = process_tweets(tweets)
        print("processed tweets")
        global dataframe_holder
        dataframe_holder = tweets
        fig = plot_polarity_subjectivity(tweets)
        print("The figure is plotted")
        return html.Div([html.H3("Sentiment Graph", style={"text-align":"center"}),
                         html.Br(),
                         dcc.Graph(id="search_graph_output", figure=fig), 
                         html.Br(),
                         html.H3("Tweets datatable", style={"text-align":"center"}),
                         html.Br(),
                         dash_table.DataTable(data=dataframe_holder[["tweet", "time_stamp", "polarity", "subjectivity"]].to_dict('records'),
                                              columns=[{'id': c, 'name': c} for c in dataframe_holder[["tweet", "time_stamp", "polarity", "subjectivity"]].columns],
                                              style_cell={
                                                'overflow': 'hidden',
                                                'textOverflow': 'ellipsis',
                                                'maxWidth': 0,
                                              },
                                              tooltip_data=[
                                                    {
                                                        column: {'value': str(value), 'type': 'markdown'}
                                                        for column, value in row.items()
                                                    } for row in dataframe_holder[["tweet", "time_stamp", "polarity", "subjectivity"]].to_dict('records')
                                              ],
                                              tooltip_duration=None,
                                              page_size=10)])

@app.callback(
    Output("search_graph_output", "figure"),
    Input("search_choice", "value"),
)

def update_search_graph_output(value):
    fig = plot_polarity_subjectivity(dataframe_holder, value)
    return fig


# In[19]:


app.run_server()


# In[ ]:




