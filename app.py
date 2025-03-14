import streamlit as st
import tweepy
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from PIL import Image
# Import Tensorflow dependencies
import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.models import load_model
# Immport nltk dependencies
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# Import API keys
from config_1 import consumerKey
from config_1 import consumerSecret
from config_1 import accessToken
from config_1 import accessTokenSecret

# Load the model
model = load_model("Model/tweeter_ml_trained_1.6.h5")

# st.title("Streamlit title -Twitter sentiment analysis")
#Customising Streamlit title with blue background:
html_temp = """
    <div style="background-color:#1da1f2;padding:10px">
    <h1 style="color:white;text-align:center;"> Twitter sentiment analysis</h1>
    </div>
    """
st.markdown(html_temp, unsafe_allow_html=True)

#Streamlit - Display logo on Sidebar
logo = Image.open("img/logo.png")
st.sidebar.image(logo)

#Streamlit Subheader
#Customising Streamlit subtitle in blue:
html_subtitle4 = """
    <div style="background-color:white">
    <h3 style="color:#1da1f2;text-align:left;"> Search Twitter</h3>
    </div>
    """
st.markdown(html_subtitle4, unsafe_allow_html=True)

# Streamlit User input #
text_input = st.text_input("Enter tweet handle with @ or #.")

####   Tweepy API   ##########
# Create the authentication object
authenticate = tweepy.OAuthHandler(consumerKey, consumerSecret)

# Set the access token and access token secret
authenticate.set_access_token(accessToken, accessTokenSecret)

# Creating the API object while passing in the auth information
api = tweepy.API(authenticate, wait_on_rate_limit=True)

# To ensure Retweets are excluded add -RT to the search term:
tweet_handle = text_input+" -RT"

if tweet_handle == "":
    posts = ""
else:
    posts = api.search(
        q=tweet_handle, retweeted="False", result_type='recent', count=100, lang="en", tweet_mode="extended")


# Create a function to preprocess the tweets to fit our ML model:

def preprocessing(messages):
    # Initialise PorterStemmer for Stemming
    ps = PorterStemmer()
# Create an empty list named corpus that will contain our cleaned sentences and words
    corpus = []
# Create a loop to clean all the text in messages:
    for i in range(0, len(messages)):
        try:
        # use re (regular expressions) to substitute all characters except [a-zA-Z] by blank in message 'text'
            review = re.sub('[^a-zA-Z]', ' ', messages[i])
    # convert all the characters as lower case
            review = review.lower()
    # split all the words in each sentence to be able to later remove the stopwords
            review = review.split()

    # create a loop in review: for each word in review, keep only words that are not stopwords list and apply 'Stemming'
            review = [ps.stem(word)
                  for word in review if not word in stopwords.words('english')]
    # join words with a space to build the review
            review = ' '.join(review)
    # append the review into the corpus
            corpus.append(review)
        except KeyError:
            corpus.append("Oops, message not retrieved!")
            pass
        
    # One_hot representation
    # each word in the corpus is allocated a number within the sentence.
    voc_size = 10000
    onehot_repr = [one_hot(words, voc_size)for words in corpus]
    # Word embedding
    sent_length = 50
    # Embebbed each sentence as a matrix
    embedded_docs = pad_sequences(
        onehot_repr, padding='pre', maxlen=sent_length)
    # Storing embedded_docs into an array
    X_final = np.array(embedded_docs)
    return(X_final)


def getSentiment(array):
    messages = array
    # Preprocessing tweets to fit the model
    X_final = preprocessing(messages)
    # Predict y values on X_final
    y_pred = model.predict_classes(X_final)

    return y_pred


def getAnalysis(score):
    if score == 1:
        return "Positive"
    else:
        return "Negative"

# Create a dataframe with a column called Tweets
def get_data(user_name):
    df = pd.DataFrame([tweet.full_text for tweet in posts], columns=['Tweets'])
    # Drop Null values
    df = df.dropna()
    # Drop duplicates
    df = df.drop_duplicates()
    # Create a new column 'Sentiment' with y values predicted
    X = df['Tweets']
    y = getSentiment(X)
    df['Sentiment'] = y
    df["Sentiment"] = df['Sentiment'].apply(getAnalysis)
    
        
    return df

# Create a function to clean the tweets:


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

# Streamlit - Creating a button to fetch the five recent tweets:
#Customising Streamlit subtitle in blue:
html_subtitle2 = """
    <div style="background-color:white">
    <h3 style="color:#1da1f2;text-align:left;"> Check the last five tweets</h3>
    </div>
    """
st.markdown(html_subtitle2, unsafe_allow_html=True)

#Action when Streamlit button is pressed:
if st.button("Recent Tweets"):
    st.write("Show the five recent tweets")
    i = 1
    for tweet in posts[0:5]:
        st.write(str(i) + '- ' + tweet.full_text + "\n")
        i = i+1

# Create a Word Cloud

def Word_Cloud(df_column):
    df_column = df_column.apply(cleanTxt)
    # Join all the tweets in Df["Tweets"] by a space
    Words = " ".join([tweets for tweets in df_column])
    wordcloud = WordCloud(width=640, height=480, random_state=21,
                          max_font_size=110, background_color="white").generate(Words)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    img = wordcloud.to_file("img/word_cloud.png")
    img= Image.open("img/word_cloud.png")

    return img

def bar_chart():
    sentiment_df = df.groupby(["Sentiment"]).count()
    #Create bargraph
    fig = sentiment_df.plot(kind='bar', stacked=True, color= ["#1da1f2"], legend=False, rot=0)
    # Set textual properties
    plt.title("Sentiment Analysis of Tweets")
    plt.ylabel("Number of Tweets")
    plt.xlabel(" ")
    #Save plot
    img2 = plt.savefig("img/bar.png")
    img2 = Image.open("img/bar.png")
    return img2


# Steamlit - Creating a button to show the sentiment analysis and word cloud
#Customising Streamlit subtitle in blue:
html_subtitle3 = """
    <div style="background-color:white">
    <h3 style="color:#1da1f2;text-align:left;"> Get Tweets' sentiment and word cloud</h3>
    </div>
    """
st.markdown(html_subtitle3, unsafe_allow_html=True)
#Display charts when button pressed:
if st.button("Sentiment Analysis"):
    df = get_data(tweet_handle)
    chart = bar_chart()
    wordcloud = Word_Cloud(df["Tweets"])
    st.image([chart,wordcloud])

# Steamlit - Creating a button to show the tweets in a dataframe
#Customising Streamlit subtitle in blue:
html_subtitle4 = """
    <div style="background-color:white">
    <h3 style="color:#1da1f2;text-align:left;"> Check the data </h3>
    </div>
    """
st.markdown(html_subtitle4, unsafe_allow_html=True)
#Render dataframe when button pressed:
if st.button("Show Data"):
    df = get_data(tweet_handle)
    st.table(df)


