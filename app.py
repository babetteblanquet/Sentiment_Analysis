import streamlit as st
import tweepy
# from wordcloud import WordCloud
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
# Import API key
from config import consumerKey
from config import consumerSecret
from config import accessToken
from config import accessTokenSecret

# Create the authentication object
authenticate = tweepy.OAuthHandler(consumerKey, consumerSecret)

# Set the access token and access token secret
authenticate.set_access_token(accessToken, accessTokenSecret)

# Creating the API object while passing in the auth information
api = tweepy.API(authenticate, wait_on_rate_limit=True)


# st.title("Streamlit example")
html_temp = """
    <div style="background-color:#1da1f2;padding:10px">
    <h1 style="color:white;text-align:center;"> Twitter sentiment analysis</h1>
    </div>
    """
st.markdown(html_temp, unsafe_allow_html=True)

st.title("Get the 100 latest tweets")
st.subheader("""Get the 100 latest tweets""")
st.write("Get the 100 latest tweets")
text_input = st.text_input("Enter tweet handle with @ or #.")
#To ensure Retweets are excluded add -RT to the search term:
tweet_handle = text_input+" -RT"

if tweet_handle =="":
    posts = ""
else:
    posts = api.search(
    q=tweet_handle, retweeted = "False", result_type='recent', count=100, lang="en", tweet_mode="extended")



#Create a dataframe with a column called Tweets
def get_data(user_name):

    df = pd.DataFrame( [tweet.full_text for tweet in posts], columns=['Tweets'])
    return df

#Clean the text

#Create a function to clean the tweets:
def cleanTxt(text):
    # Removing @mentions
    text = re.sub(r'@[A-Za-z0–9]+', '', text)
    # Removing '#' hash tag symbol
    text = re.sub(r'#', '', text)
    # Removing RT re-tweet
    text = re.sub(r'RT[\s]+', '', text)
    # Removing hyperlink
    text = re.sub(r'https?:\/\/\S+', '', text)
    return text


# Creating a button to show the tweets in a dataframe
if st.button("Show Data"):
    st.success("Fetching Last 100 Tweets")
    df = get_data(tweet_handle)
    #Clean the Tweets
    df['Tweets'] = df['Tweets'].apply(cleanTxt)
    st.write(df)


# Creating a button to fetch the recent five tweets:
if st.button("Recent Tweets"):
    st.write("Show the five recent tweets")
    i = 1
    for tweet in posts[0:5]:
        st.write(str(i) + '- ' + tweet.full_text + "\n")
        i = i+1

#Create a function to get the subjectivity
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

    # 			# Create a function to get the polarity
    # def getPolarity(text):
    # 	return  TextBlob(text).sentiment.polarity

    # Create two new columns 'Subjectivity' & 'Polarity'
    df['Subjectivity'] = df['Tweets'].apply(getSubjectivity)
    # df['Polarity'] = df['Tweets'].apply(getPolarity)

    # def getAnalysis(score):
    # 	if score < 0:
    # 		return 'Negative'

    # 	elif score == 0:
    # 		return 'Neutral'

    # 	else:
    # 		return 'Positive'

    # df['Analysis'] = df['Polarity'].apply(getAnalysis)


# Call the sentiment analysis algorythm
# Load the model
# from tensorflow.keras.models import load_model
# model = load_model("mnist_trained.h5")

# def getSubjectivity(text):
# 				return TextBlob(text).sentiment.subjectivity


# if __name__=='__main__':
#     main()
