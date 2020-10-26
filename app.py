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
tweet_handle = st.text_input("Enter tweet handle with @ or #.")

if tweet_handle =="":
    posts = ""
else:
    posts = api.search(
    q=tweet_handle, result_type='recent', count=100, lang="en", tweet_mode="extended")


def cleanTxt(text):
    text = re.sub('@[A-Za-z0–9]+', '', text)  # Removing @mentions
    text = re.sub('#', '', text)  # Removing '#' hash tag
    text = re.sub('RT[\s]+', '', text)  # Removing RT
    text = re.sub('https?:\/\/\S+', '', text)  # Removing hyperlink
    return text


def get_data(user_name):

    df = pd.DataFrame(
        [tweet.full_text for tweet in posts], columns=['Tweets'])
    return df


# Creating a button to fetch all the tweets
if st.button("Show Data"):
    st.success("Fetching Last 100 Tweets")
    df = get_data(tweet_handle)
    st.write(df)


# Creating a button to fetch the recent five tweets:
if st.button("Recent Tweets"):
    st.success("Show the five recent tweets")
    for tweet in posts[0:5]:
        st.write(tweet.full_text + "\n")

    # Clean the tweets
    # df['Tweets'] = df['Tweets'].apply(cleanTxt)

    # def getSubjectivity(text):
    # 	return TextBlob(text).sentiment.subjectivity

    # 			# Create a function to get the polarity
    # def getPolarity(text):
    # 	return  TextBlob(text).sentiment.polarity

    # 			# Create two new columns 'Subjectivity' & 'Polarity'
    # df['Subjectivity'] = df['Tweets'].apply(getSubjectivity)
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
