# sentiment-analysis-twitterAPI
This is a short tutorial to show how I pull tweets from Twitter API on a precise (or various) topic and use it for a sentiment analysis.
In this case, we have 3 main files:
1. pull_tweets.py : gather our tweets from Twitter API using account tokens and keys
2. TrumpVSBiden.py : Sentiment Analysis on the topic "Trump VS Biden" with visualization
3. SentimentAnalysis-Dashboard.py : A dashboard with the previous visualization and **most importantly a sentiment analysis on any topic typed by the user**

## Set up
### Requirements
Install the requirements on *requirements.txt* by entering the following command on the terminal
```
pip install -r requirements.txt
```

### Twitter Developer Account
You also have to create a Twitter Developer account which is very important so that you can pull tweets with the keys and tokens that Twitter provide you.
Here, you will use it on *pull_tweets.py* and *SentimentAnalysis-Dashboard* files where you will need :
* Access Token
* Access Token Secret
* Consumer Key
* Consumer Secret
Here is the link to apply for a Twitter Developer Account: : https://developer.twitter.com/en/apply-for-access
  
It may take time (around one week to have the account after the appliance)

### Pulling tweets
Now, after having all the requirements needed and a Twitter Developer account, let's jump on the code !
Copy-Paste your different keys/tokens on the concerned fields and choose your topics in the final line.
After that, execute the code with the following command on the terminal to obtain our output on a *twitter_data.txt* file:
```
python pull_tweets.py > twitter_data.txt
```
You have to wait now so that all your tweets and their informations are gathered.

## Visualization and Dashboard
### Visualization
You can now visualize the sentiment analysis of your tweets by using the same code on *TrumpVSBiden.py* file.

### Dashboard
Here, we will talk about the *SentimentAnalysis-Dashboard.py* file where all the things done before are reunited on one dashboard (Flask Application) + the real-time visualization on a topic typed by the user
You also have to copy-paste your Twitter Developer info in this file.

## Deployment
Unfortunately, the deployment on Heroku doesn't work (I think because of the *ntlk* large memory).
So we can just use the deployment on the server of the Flask Application.

![img.png](img.png)
![img_1.png](img_1.png)

### Example of sentiment analysis on a topic typed by the user

![img_2.png](img_2.png)
![img_3.png](img_3.png)