#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[42]:


df = pd.read_csv("sample_50000.csv")
df


# In[43]:


df.drop("Unnamed: 0",   axis='columns', inplace=True)
df


# In[44]:


##Get the independent features
X = df.drop("sentiment", axis=1)


# In[45]:


## Get the Dependent features
y=df["sentiment"]


# In[46]:


y.value_counts()


# In[47]:


X.shape


# In[48]:


y.shape


# In[49]:


import tensorflow as tf


# In[50]:


tf.__version__


# In[51]:


from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense


# In[52]:


### Vocabulary size
voc_size=10000


# ## Data Preprocessing

# In[53]:


#Creating a copy of X into the variable messages
messages = X.copy()


# In[54]:


messages['text'][0]


# In[55]:


#Reset index as we drop Nan values from df
messages.reset_index(inplace=True)


# In[56]:


import nltk
import re
from nltk.corpus import stopwords


# In[57]:


#In order to remove words that are not meanningful (e.g. the, a, then, often...), we need to download those words.
nltk.download("stopwords")


# In[58]:


##Data Preprocessing/Cleaning
from nltk.stem.porter import PorterStemmer
#Initialise PorterStemmer for Stemming
ps = PorterStemmer()
#Create an empty list named corpus that will contain our cleaned sentences and words
corpus = []
#Create a loop to clean all the text in messages:
for i in range(0, len(messages)):
    #print index
    print(i)
    #use re (regular expressions) to substitute all characters except [a-zA-Z] by blank in message 'text'
    review = re.sub('[^a-zA-Z]', ' ', messages['text'][i])
    #convert all the characters as lower case
    review = review.lower()
    #split all the words in each sentence to be able to later remove the stopwords
    review = review.split()
    
    #create a loop in review: for each word in review, keep only words that are not stopwords list and apply 'Stemming'
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    #join words with a space to build the review
    review = ' '.join(review)
    #append the review into the corpus
    corpus.append(review)


# In[59]:


#Checking the new sentences in the corpus
for i in range(0,5):
    print(corpus[i])


# ## One-Hot representation of words/sentences

# In[60]:


#Apply One_hot representation for each word in the corpus based on the voc_size - each word is allocated a number within the sentence.
onehot_repr=[one_hot(words,voc_size)for words in corpus] 
onehot_repr


# In[81]:


#Checking the maximum length of all sentences.
number_words=[]
for i in range(0,len(messages)):
    number_words.append(len(onehot_repr[i]))


# In[82]:


import numpy as np


# In[83]:


max_value = np.max(number_words)
max_value


# ## Word Embedding

# In[87]:


#Use pad sequencing to ensure all sentences are the same length.
#Set up the common length of each sentence. We used the max_value of number of words
def word_embedding(onehot_repr):
    sent_length=31
    #Embebbed each sentence as a matrix
    embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
    return(embedded_docs)


# In[88]:


word_embedding(onehot_repr)


# In[65]:


#Check first sentence embedded with 100 words
embedded_docs[0]


# ## Creating the model Sequential

# In[66]:


##Creating model
#Determining the number of features we want for each vector
embedding_vector_features=10
#Initialise model Sequential
model=Sequential()
#Adding embedding layer: vocabulary size, number of vector features, number of words per sentence
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
#Adding the LSTM layer (Long Short Term Memory) of 100 neurons
model.add(LSTM(20))
#Add extra layer
#model.add(Dense(100, activation='relu'))
#Adding the Dense layer with sigmoid activation
model.add(Dense(1,activation='sigmoid'))
#Compiling the model
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())


# In[67]:


#Checking the shape of embedded_docs
len(embedded_docs),y.shape


# In[68]:


#Storing embedded_docs into an array
X_final = np.array(embedded_docs)
y_final = np.array(y)


# In[84]:


X_final


# In[69]:


X_final.shape, y_final.shape


# In[70]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42)


# ## Model Training

# In[71]:


#Training the model
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=5,batch_size=64)


# ## Adding Drop out

# In[72]:


# from tensorflow.keras.layers import Dropout
# ## Creating model
# embedding_vector_features=40
# model=Sequential()
# model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
# model.add(Dropout(0.3))
# model.add(LSTM(100))
# model.add(Dropout(0.3))
# model.add(Dense(1,activation='sigmoid'))
# model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[73]:


#Training the model
# model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64)


# ## Performance Metrics And Accuracy

# In[74]:


y_pred=model.predict_classes(X_test)


# In[75]:


from sklearn.metrics import confusion_matrix


# In[76]:


# import matplotlib.pyplot as plt


# In[77]:


confusion_matrix(y_test,y_pred)


# In[78]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[79]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[80]:


# Save the model
model.save("tweeter_ml_trained_50000.h5")


# In[ ]:




