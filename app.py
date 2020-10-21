import streamlit as st
import tweepy
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt



#st.title("Streamlit example")
html_temp = """
    <div style="background-color:#1da1f2;padding:10px">
    <h1 style="color:white;text-align:center;"> Twitter sentiment analysis</h1>
    </div>
    """
st.markdown(html_temp,unsafe_allow_html=True)

st.title("Get the 100 latest tweets")
st.subheader("""Get the 100 latest tweets""")
st.write("Get the 100 latest tweets")
tweet_handle = st.text_input("Enter tweet handle without the @")

def cleanTxt(text):
	text = re.sub('@[A-Za-z0–9]+', '', text) #Removing @mentions
	text = re.sub('#', '', text) # Removing '#' hash tag
	text = re.sub('RT[\s]+', '', text) # Removing RT
	text = re.sub('https?:\/\/\S+', '', text) # Removing hyperlink
	return text	

def get_data(user_name):

	tweets = api.user_timeline(screen_name=tweet_handle, count = 100, lang ="en", tweet_mode="extended")
	df = pd.DataFrame([tweet.full_text for tweet in tweets], columns=['Tweets'])

    # Clean the tweets
	df['Tweets'] = df['Tweets'].apply(cleanTxt)
    return df
           

## Call the sentiment analysis algorythm
# Load the model
# from tensorflow.keras.models import load_model
# model = load_model("mnist_trained.h5")

# def getSubjectivity(text):
# 				return TextBlob(text).sentiment.subjectivity
    

#Button show data
if st.button("Show Tweets"):

			st.success("Fetching Last 100 Tweets")
			df=get_data(tweet_handle)
			st.write(df)


# if __name__=='__main__':
#     main()