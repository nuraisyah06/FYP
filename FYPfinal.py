#!/usr/bin/env python
# coding: utf-8

#Import library: 
import streamlit as st
import os
import googleapiclient.discovery
from tabulate import tabulate
from pytube import YouTube
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image
import io
from io import BytesIO
import requests
import locale
from wordcloud import WordCloud
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import random


stop_words = stopwords.words()

# # Get URL Input *********************************************************************************************************************
st.set_page_config(page_title = 'YouTube Comment Analyzer',
                    layout='wide',
                    initial_sidebar_state='collapsed')

header_left,header_mid,header_right = st.columns([0.75,2,0.5],gap='medium')

with header_mid:
    st.title('YouTube Comment Analyzer')
    st.write('')
    st.write('')
    st.write('')

# Set API key **************************************************************************************************************************
api_key = "AIzaSyAM5q0GBVwwW8j5IlFXd59tS5eFUx5N6T4"

# Create YouTube Data API client
youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)


# Prompt user for YouTube video URL ****************************************************************************************************
Q1,Q2 = st.columns(2) # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
with Q1: 
    link_URL_1 = st.text_input('Enter your 1st YouTbe video URL:')
    
with Q2: 
    link_URL_2 = st.text_input('Enter your 2nd YouTbe video URL:')
    
# Extract video ID from the URL *********************************************************************************************************

video_id_1 = link_URL_1.split("v=")[1]
video_id_1 = os.path.basename(video_id_1)

video_id_2 = link_URL_2.split("v=")[1]
video_id_2 = os.path.basename(video_id_2)

# Get YouTube comments function *********************************************************************************************************
def get_video_comments1(video_id_1): # -------------------------------------------------------
    comments1 = []
    next_page_token = None

    while True:
        # Make API request to retrieve comments
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id_1,
            textFormat="plainText",
            pageToken=next_page_token,
            maxResults=100
        ).execute()

        # Extract comments and add them to the list
        for item in response["items"]:
            comment1 = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments1.append(comment1)

        # Check if there are more comments and retrieve the next page token
        if "nextPageToken" in response:
            next_page_token = response["nextPageToken"]
        else:
            break

    return comments1

def get_video_comments2(video_id_2): # -------------------------------------------------------
    comments2 = []
    next_page_token = None

    while True:
        # Make API request to retrieve comments
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id_2,
            textFormat="plainText",
            pageToken=next_page_token,
            maxResults=100
        ).execute()

        # Extract comments and add them to the list
        for item in response["items"]:
            comment2 = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments2.append(comment2)

        # Check if there are more comments and retrieve the next page token
        if "nextPageToken" in response:
            next_page_token = response["nextPageToken"]
        else:
            break

    return comments2


Q3,Q4 = st.columns(2) # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Function Takes YouTube Object as Argument. **************************************************

def video_info1(yt1):
    with Q3: 
        #---------------------------------------------------1st Video Thumbnail-------------------------
        # Retrieve the thumbnail URL
        thumbnail_url_1 = yt1.thumbnail_url
        # Download the image from the URL
        response1 = requests.get(thumbnail_url_1)
        image1 = Image.open(BytesIO(response1.content))
        # Resize the image
        resized_image1 = image1.resize((400, 250))  # Specify the desired width and height
        # Display the resized thumbnail image
        st.image(resized_image1, caption='Video Thumbnail')
      #---------------------------------------------------Video Title-----------------------------
        st.write(f'You entered video URL of : {yt1.title}')
      #---------------------------------------------------Video Duration--------------------------
        duration_minutes, duration_seconds = divmod(yt1.length, 60)
        st.write(f'Video duration: {duration_minutes:02d}:{duration_seconds:02d}')
      #---------------------------------------------------Video Views-----------------------------
        # Set the locale to the user's default
        locale.setlocale(locale.LC_ALL, '')
        # Retrieve the video views
        video_views1 = yt1.views
        # Format the views with thousands separators
        formatted_views1 = locale.format_string("%d", video_views1, grouping=True)
        # Display the formatted video views
        st.write(f'Video views: {formatted_views1} views')
        
def video_info2(yt2):
    with Q4: 
        #---------------------------------------------------2nd Video Thumbnail-------------------------
        # Retrieve the thumbnail URL
        thumbnail_url_2 = yt2.thumbnail_url
        # Download the image from the URL
        response2 = requests.get(thumbnail_url_2)
        image2 = Image.open(BytesIO(response2.content))
        # Resize the image
        resized_image2 = image2.resize((400, 250))  # Specify the desired width and height
        # Display the resized thumbnail image
        st.image(resized_image2, caption='Video Thumbnail')
        
      #---------------------------------------------------Video Title-----------------------------
        st.write(f'You entered video URL of : {yt2.title}')
      #---------------------------------------------------Video Duration--------------------------
        duration_minutes, duration_seconds = divmod(yt2.length, 60)
        st.write(f'Video duration: {duration_minutes:02d}:{duration_seconds:02d}')
      #---------------------------------------------------Video Views-----------------------------
        # Set the locale to the user's default
        locale.setlocale(locale.LC_ALL, '')
        # Retrieve the video views
        video_views2 = yt2.views
        # Format the views with thousands separators
        formatted_views2 = locale.format_string("%d", video_views2, grouping=True)
        # Display the formatted video views
        st.write(f'Video views: {formatted_views2} views')

# Create Youtube Object ------------------------------------
yt1 = YouTube(link_URL_1) 
yt2 = YouTube(link_URL_2) 

# call the function to get video metadata -------------------
video_info1(yt1)
video_info2(yt2)

# Call the function to get video comments -------------------
comments1 = get_video_comments1(video_id_1)
comments2 = get_video_comments2(video_id_2)

# Create a DataFrame from comments --------------------------
df1 = pd.DataFrame(comments1, columns=["Comment"])
df2 = pd.DataFrame(comments2, columns=["Comment"])

# # Load Data

df1.rename(columns={'Comment':'text1'}, inplace = True)
df2.rename(columns={'Comment':'text2'}, inplace = True)

# # Cleaning ***************************************************************************************************************

def cleaning1(text1): # -----------------------------------------------------------------------------------------------------------
    # converting to lowercase, removing URL links, special characters, punctuations...
    text1 = text1.lower() # converting to lowercase
    text1 = re.sub('https?://\S+|www\.\S+', '', text1) # removing URL links
    text1 = re.sub(r"\b\d+\b", "", text1) # removing number 
    text1 = re.sub('<.*?>+', '', text1) # removing special characters, 
    text1 = re.sub('[%s]' % re.escape(string.punctuation), '', text1) # punctuations
    text1 = re.sub('\n', '', text1)
    text1 = re.sub('[’“”…]', '', text1)
   
    #removing emoji: 
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    text1 = emoji_pattern.sub(r'', text1)  

   # removing short form: 
    text1=re.sub("isn't",'is not',text1)
    text1=re.sub("he's",'he is',text1)
    text1=re.sub("wasn't",'was not',text1)
    text1=re.sub("there's",'there is',text1)
    text1=re.sub("couldn't",'could not',text1)
    text1=re.sub("won't",'will not',text1)
    text1=re.sub("they're",'they are',text1)
    text1=re.sub("she's",'she is',text1)
    text1=re.sub("There's",'there is',text1)
    text1=re.sub("wouldn't",'would not',text1)
    text1=re.sub("haven't",'have not',text1)
    text1=re.sub("That's",'That is',text1)
    text1=re.sub("you've",'you have',text1)
    text1=re.sub("He's",'He is',text1)
    text1=re.sub("what's",'what is',text1)
    text1=re.sub("weren't",'were not',text1)
    text1=re.sub("we're",'we are',text1)
    text1=re.sub("hasn't",'has not',text1)
    text1=re.sub("you'd",'you would',text1)
    text1=re.sub("shouldn't",'should not',text1)
    text1=re.sub("let's",'let us',text1)
    text1=re.sub("they've",'they have',text1)
    text1=re.sub("You'll",'You will',text1)
    text1=re.sub("i'm",'i am',text1)
    text1=re.sub("we've",'we have',text1)
    text1=re.sub("it's",'it is',text1)
    text1=re.sub("don't",'do not',text1)
    text1=re.sub("that´s",'that is',text1)
    text1=re.sub("I´m",'I am',text1)
    text1=re.sub("it’s",'it is',text1)
    text1=re.sub("she´s",'she is',text1)
    text1=re.sub("he’s'",'he is',text1)
    text1=re.sub('I’m','I am',text1)
    text1=re.sub('I’d','I did',text1)
    text1=re.sub("he’s'",'he is',text1)
    text1=re.sub('there’s','there is',text1)
    text1=re.sub('thx','thanks',text1)
    
     
    return text1
    
dt1 = df1['text1'].apply(cleaning1)


def cleaning2(text2): # -----------------------------------------------------------------------------------------------------------
    # converting to lowercase, removing URL links, special characters, punctuations...
    text2 = text2.lower() # converting to lowercase
    text2 = re.sub('https?://\S+|www\.\S+', '', text2) # removing URL links
    text2 = re.sub(r"\b\d+\b", "", text2) # removing number 
    text2 = re.sub('<.*?>+', '', text2) # removing special characters, 
    text2 = re.sub('[%s]' % re.escape(string.punctuation), '', text2) # punctuations
    text2 = re.sub('\n', '', text2)
    text2 = re.sub('[’“”…]', '', text2)
   
    #removing emoji: 
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    text2 = emoji_pattern.sub(r'', text2)  

   # removing short form: 
    text2=re.sub("isn't",'is not',text2)
    text2=re.sub("he's",'he is',text2)
    text2=re.sub("wasn't",'was not',text2)
    text2=re.sub("there's",'there is',text2)
    text2=re.sub("couldn't",'could not',text2)
    text2=re.sub("won't",'will not',text2)
    text2=re.sub("they're",'they are',text2)
    text2=re.sub("she's",'she is',text2)
    text2=re.sub("There's",'there is',text2)
    text2=re.sub("wouldn't",'would not',text2)
    text2=re.sub("haven't",'have not',text2)
    text2=re.sub("That's",'That is',text2)
    text2=re.sub("you've",'you have',text2)
    text2=re.sub("He's",'He is',text2)
    text2=re.sub("what's",'what is',text2)
    text2=re.sub("weren't",'were not',text2)
    text2=re.sub("we're",'we are',text2)
    text2=re.sub("hasn't",'has not',text2)
    text2=re.sub("you'd",'you would',text2)
    text2=re.sub("shouldn't",'should not',text2)
    text2=re.sub("let's",'let us',text2)
    text2=re.sub("they've",'they have',text2)
    text2=re.sub("You'll",'You will',text2)
    text2=re.sub("i'm",'i am',text2)
    text2=re.sub("we've",'we have',text2)
    text2=re.sub("it's",'it is',text2)
    text2=re.sub("don't",'do not',text2)
    text2=re.sub("that´s",'that is',text2)
    text2=re.sub("I´m",'I am',text2)
    text2=re.sub("it’s",'it is',text2)
    text2=re.sub("she´s",'she is',text2)
    text2=re.sub("he’s'",'he is',text2)
    text2=re.sub('I’m','I am',text2)
    text2=re.sub('I’d','I did',text2)
    text2=re.sub("he’s'",'he is',text2)
    text2=re.sub('there’s','there is',text2)
    text2=re.sub('thx','thanks',text2)
    
     
    return text2
    
dt2 = df2['text2'].apply(cleaning2)

dt1 = pd.DataFrame(dt1)
dt2 = pd.DataFrame(dt2)

# remove stop word: **********************************************************************************************************

dt1['no_sw'] = dt1['text1'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
dt2['no_sw'] = dt2['text2'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

#Working with the most Frequent Words: ***************************************************************************************
from collections import Counter
 ## 1 ------------------------------------------
cnt1 = Counter()
for text1 in dt1["no_sw"].values:
    for word in text1.split():
        cnt1[word] += 1
cnt1.most_common(10)
temp1 = pd.DataFrame(cnt1.most_common(10))
temp1.columns=['word', 'count']

 ## 2 ------------------------------------------
cnt2 = Counter()
for text2 in dt2["no_sw"].values:
    for word in text2.split():
        cnt2[word] += 1
cnt2.most_common(10)
temp2 = pd.DataFrame(cnt2.most_common(10))
temp2.columns=['word', 'count']

# Insert a divider between rows
st.markdown('<hr style="height: 2px; border-width: 0; background-color: #808080; margin: 20px 0;">', unsafe_allow_html=True)

Q5,Q6 = st.columns(2) # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

with Q5: # #1----------------------------------------------------------------------------------------------------#
    # Create the bar chart with hot pink color gradient
    image1_barchart = px.bar(temp1, x="count", y="word", title='Top 10 Most Frequent Words in Comments of Video 1', orientation='h',
                         color="count", color_continuous_scale=['pink', 'red'],
                         text='count', labels={'count': 'Frequency'}, width=800, height=600)

    # Update the layout to add a background color and set the title properties
    image1_barchart.update_layout(
    title_text='Top 10 Most Frequent Words in Comments of Video 1',
    title_x=0.5,
    title_y=0.98,
    title_xanchor='center',
    title_yanchor='top',
    title_font=dict(size=24, color='black'),
    plot_bgcolor='white', 
    paper_bgcolor='rgba(0,0,0,0)'
    )

    # Save the plot to a BytesIO object
    image1_stream = io.BytesIO()
    image1_barchart.write_image(image1_stream, format='PNG')

    # Open the image from the BytesIO object
    image1 = Image.open(image1_stream)

    # Display the image using Streamlit
    st.image(image1, use_column_width=True)

    
with Q6: # #2----------------------------------------------------------------------------------------------------#
    # Create the bar chart with hot pink color gradient
    image2_barchart = px.bar(temp1, x="count", y="word", title='Top 10 Most Frequent Words in Comments of Video 2', orientation='h',
                         color="count", color_continuous_scale=['pink', 'red'],
                         text='count', labels={'count': 'Frequency'}, width=800, height=600)

    # Update the layout to add a background color and set the title properties
    image1_barchart.update_layout(
    title_text='Top 10 Most Frequent Words in Comments of Video 2',
    title_x=0.5,
    title_y=0.98,
    title_xanchor='center',
    title_yanchor='top',
    title_font=dict(size=24, color='black'),
    plot_bgcolor='white', 
    paper_bgcolor='rgba(0,0,0,0)'
    )

    # Save the plot to a BytesIO object
    image2_stream = io.BytesIO()
    image2_barchart.write_image(image2_stream, format='PNG')

    # Open the image from the BytesIO object
    image2 = Image.open(image2_stream)

    # Display the image using Streamlit
    st.image(image2, use_column_width=True)

# Remove the most frequent words: ************************************************************************************************
### #1 ------------------------------------------------------------------------------------
FREQWORDS1= set([w for (w, wc) in cnt1.most_common(10)])
def remove_freqwords1(text1):
    """custom function to remove the frequent words"""
    return " ".join([word for word in str(text1).split() if word not in FREQWORDS1])
dt1["wo_stopfreq1"] = dt1["no_sw"].apply(lambda text1: remove_freqwords1(text1))

### #2 ------------------------------------------------------------------------------------
FREQWORDS2= set([w for (w, wc) in cnt2
                 .most_common(10)])
def remove_freqwords2(text2):
    """custom function to remove the frequent words"""
    return " ".join([word for word in str(text2).split() if word not in FREQWORDS2])
dt2["wo_stopfreq2"] = dt2["no_sw"].apply(lambda text2: remove_freqwords2(text2))


# Lemmatization: Lemmatization is converting the word to its base form or lemma by removing affixes from the inflected words. 
# It helps to create better features for machine learning and NLP models hence it is an important preprocessing step.
wordnet_lem = WordNetLemmatizer()

dt1['wo_stopfreq_lem1'] = dt1['wo_stopfreq1'].apply(wordnet_lem.lemmatize)
dt2['wo_stopfreq_lem2'] = dt2['wo_stopfreq2'].apply(wordnet_lem.lemmatize)

# # Sentiment Analysis on the YouTube Comments Dataset *********************************************************************************

import nltk
#nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

dt1['Sentiment Scores'] = dt1['text1'].apply(lambda x:sia.polarity_scores(x)['compound'])
dt1['Sentiment'] = dt1['Sentiment Scores'].apply(lambda s : 'Positive' if s > 0 else ('Neutral' if s == 0 else 'Negative'))

dt2['Sentiment Scores'] = dt2['text2'].apply(lambda x:sia.polarity_scores(x)['compound'])
dt2['Sentiment'] = dt2['Sentiment Scores'].apply(lambda s : 'Positive' if s > 0 else ('Neutral' if s == 0 else 'Negative'))

# # Calculate percentage of sentiment **************************************************************************************************

#comments polarity ## #1------------------------------------------------------------
videos1 = []
for i in range(0,len(dt1)):
    aP1 = len(dt1[(dt1.Sentiment == 'Positive')])
    bP1 = len(dt1)
    PercentageP1 = (aP1/bP1)*100
    videos1.append(round(PercentageP1,2))
    
    aN1 = len(dt1[(dt1.Sentiment == 'Negative')])
    bN1 = len(dt1)
    PercentageN1 = (aN1/bN1)*100
    videos1.append(round(PercentageN1,2))
    
    a1 = len(dt1[(dt1.Sentiment == 'Neutral')])
    b1 = len(dt1)
    Percentage1 = (a1/b1)*100
    videos1.append(round(Percentage1,2))

#comments polarity ## #2------------------------------------------------------------
videos2 = []
for i in range(0,len(dt2)):
    aP2 = len(dt2[(dt2.Sentiment == 'Positive')])
    bP2 = len(dt2)
    PercentageP2 = (aP2/bP2)*100
    videos2.append(round(PercentageP2,2))
    
    aN2 = len(dt2[(dt2.Sentiment == 'Negative')])
    bN2 = len(dt2)
    PercentageN2 = (aN2/bN2)*100
    videos2.append(round(PercentageN2,2))
    
    a2 = len(dt2[(dt2.Sentiment == 'Neutral')])
    b2 = len(dt2)
    Percentage2 = (a2/b2)*100
    videos2.append(round(Percentage2,2))

    # Insert a divider between rows
st.markdown('<hr style="height: 2px; border-width: 0; background-color: #808080; margin: 20px 0;">', unsafe_allow_html=True)

# # Piechart****************************************************************************************************************
Q7,Q8 = st.columns(2) #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import matplotlib.pyplot as plt

## #1-----------------------------------------------------------------------------------------------------------
with Q7:
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = 'Positive', 'Negative', 'Neutral'
    sizes = [PercentageP1, PercentageN1, Percentage1]
    colors = ['#FF69B4', '#FF1493', '#FFC0CB']  # Pink shades
    
    # Create the pie chart
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes,  labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
    # Add a title to the pie chart
    ax1.set_title("Percentage of Sentiment in the Comment of Video 1", fontsize=16)
    
    # Save the plot to a BytesIO object
    image3_stream = io.BytesIO()
    plt.savefig(image3_stream, format='PNG')
    plt.close()

    # Reset the BytesIO object for reading
    image3_stream.seek(0)

    # Open the image from the BytesIO object
    image3 = Image.open(image3_stream)

    # Display the image using Streamlit
    st.image(image3, use_column_width=True)

## #2-----------------------------------------------------------------------------------------------------------
with Q8:
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = 'Positive', 'Negative', 'Neutral'
    sizes = [PercentageP2, PercentageN2, Percentage2]
    colors = ['#FF69B4', '#FF1493', '#FFC0CB']  # Pink shades

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes,  labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Add a title to the pie chart
    ax1.set_title("Percentage of Sentiment in the Comment of Video 2", fontsize=16)
    
    # Save the plot to a BytesIO object
    image4_stream = io.BytesIO()
    plt.savefig(image4_stream, format='PNG')
    plt.close()

    # Reset the BytesIO object for reading
    image4_stream.seek(0)

    # Open the image from the BytesIO object
    image4 = Image.open(image4_stream)

    # Display the image using Streamlit
    st.image(image4, use_column_width=True)

# Insert a divider between rows
st.markdown('<hr style="height: 2px; border-width: 0; background-color: #808080; margin: 20px 0;">', unsafe_allow_html=True)

# # WordCloud ***************************************************************************************************************

Q9,Q10 = st.columns(2) #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^positive^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

with Q9:
    # Display the centered title using Markdown syntax
    st.markdown('<p style="text-align: center; font-size: 16px;">Wordcloud for Positive Comments</p>', unsafe_allow_html=True)
    
    text1 = " ".join(i for i in dt1[dt1['Sentiment']=='Positive']['wo_stopfreq_lem1'])
    
    wordcloudP1 = WordCloud(background_color="white").generate(text1)
    
    # Create the plot
    plt.figure(figsize=(15,10))
    plt.imshow(wordcloudP1, interpolation='bilinear')
    plt.axis("off")
    plt.title('Wordcloud for Positive Comments of Video 1')
    plt.show()

    # Save the plot to a BytesIO object
    image_stream = io.BytesIO()
    plt.savefig(image_stream, format='PNG')
    plt.close()

    # Reset the BytesIO object for reading
    image_stream.seek(0)

    # Open the image from the BytesIO object
    image = Image.open(image_stream)

    # Display the image using Streamlit
    st.image(image, use_column_width=True)
    
with Q10:
    # Display the centered title using Markdown syntax
    st.markdown('<p style="text-align: center; font-size: 16px;">Wordcloud for Positive Comments</p>', unsafe_allow_html=True)
    
    text2 = " ".join(i for i in dt2[dt2['Sentiment']=='Positive']['wo_stopfreq_lem2'])
    wordcloudP2 = WordCloud( background_color="white").generate(text2)

    plt.figure( figsize=(15,10))
    plt.imshow(wordcloudP2, interpolation='bilinear')
    plt.axis("off")
    plt.title('wordcloud for Positive Comments of Video 2')
    plt.show()

    # Save the plot to a BytesIO object
    image_stream = io.BytesIO()
    plt.savefig(image_stream, format='PNG')
    plt.close()

    # Reset the BytesIO object for reading
    image_stream.seek(0)

    # Open the image from the BytesIO object
    image = Image.open(image_stream)

    # Display the image using Streamlit
    st.image(image, use_column_width=True)

# Insert a divider between rows
st.markdown('<hr style="height: 2px; border-width: 0; background-color: #808080; margin: 20px 0;">', unsafe_allow_html=True)
    
Q11,Q12 = st.columns(2) #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^negative^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
with Q11:# Display the centered title using Markdown syntax
    st.markdown('<p style="text-align: center; font-size: 16px;">Wordcloud for Negative Comments</p>', unsafe_allow_html=True)
    
    text1 = " ".join(i for i in dt1[dt1['Sentiment']=='Negative']['wo_stopfreq_lem1'])
    wordcloudP1 = WordCloud(background_color="white").generate(text1)

    plt.figure(figsize=(15,10))
    plt.imshow(wordcloudP1, interpolation='bilinear')
    plt.axis("off")
    plt.title('wordcloud for Negative Comments of Video 1')
    plt.show()

    # Save the plot to a BytesIO object
    image_stream = io.BytesIO()
    plt.savefig(image_stream, format='PNG')
    plt.close()

    # Reset the BytesIO object for reading
    image_stream.seek(0)

    # Open the image from the BytesIO object
    image = Image.open(image_stream)

    # Display the image using Streamlit
    st.image(image, use_column_width=True)
    
with Q12:
    st.markdown('<p style="text-align: center; font-size: 16px;">Wordcloud for Negative Comments</p>', unsafe_allow_html=True)
    
    text2 = " ".join(i for i in dt2[dt2['Sentiment']=='Negative']['wo_stopfreq_lem2'])
    wordcloudP2 = WordCloud( background_color="white").generate(text2)

    plt.figure( figsize=(15,10))
    plt.imshow(wordcloudP2, interpolation='bilinear')
    plt.axis("off")
    plt.title('wordcloud for Negative Comments of Video 2')
    plt.show()

    # Save the plot to a BytesIO object
    image_stream = io.BytesIO()
    plt.savefig(image_stream, format='PNG')
    plt.close()

    # Reset the BytesIO object for reading
    image_stream.seek(0)

    # Open the image from the BytesIO object
    image = Image.open(image_stream)

    # Display the image using Streamlit
    st.image(image, use_column_width=True)
    
    
    
    
    
    
# create the cleaned data for the train-test split:****************************************************************************
nb1 = dt1.drop(columns=['text1','no_sw', 'wo_stopfreq1'])
nb1['Sentiment'] = [-1 if each == "Negative" else (1 if each == "Positive" else 0) for each in nb1['Sentiment']]

nb2 = dt2.drop(columns=['text2','no_sw', 'wo_stopfreq2'])
nb2['Sentiment'] = [-1 if each == "Negative" else (1 if each == "Positive" else 0) for each in nb2['Sentiment']]

# Tokenization ****************************************************************************************************************
tokenized_review1 = nb1['wo_stopfreq_lem1'].apply(lambda x: x.split())
tokenized_review2 = nb2['wo_stopfreq_lem2'].apply(lambda x: x.split())

# Drop empty rows from the DataFrame------------------------------------
dt1 = dt1[dt1["text1"].str.strip().astype(bool)]
dt2 = dt2[dt2["text2"].str.strip().astype(bool)]

# Reset the index-------------------------------------------------------
dt1.reset_index(drop=True, inplace=True)
dt2.reset_index(drop=True, inplace=True)

# # train-test split***********************************************************************************************************
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split

token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)

text_counts1 = cv.fit_transform(nb1['wo_stopfreq_lem1'])
X1 = text_counts1
y1 = nb1['Sentiment']
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.20,random_state=30)

text_counts2 = cv.fit_transform(nb2['wo_stopfreq_lem2'])
X2 = text_counts2
y2 = nb2['Sentiment']
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.20,random_state=30)


####################################****************NaiveBayes Modelling****************#########################################

from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
#----------------------------------------------------Mutilnominal NB model-------------------------------------------#

from sklearn.naive_bayes import MultinomialNB

MNB1 = MultinomialNB()
MNB2 = MultinomialNB()
## #1-----------------------------------------------------------------------------------*
MNB1.fit(X1_train, y1_train)

predicted = MNB1.predict(X1_test)
accuracy_score = metrics.accuracy_score(predicted, y1_test)

print('MultinominalNB model accuracy is',str('{:04.2f}'.format(accuracy_score*100))+'%')
print('------------------------------------------------')
print('Confusion Matrix:')
print(pd.DataFrame(confusion_matrix(y1_test, predicted)))
print('------------------------------------------------')
print('Classification Report:')
print(classification_report(y1_test, predicted))


## #2-----------------------------------------------------------------------------------*
MNB2.fit(X2_train, y2_train)

predicted = MNB2.predict(X2_test)
accuracy_score = metrics.accuracy_score(predicted, y2_test)

print('MultinominalNB model accuracy is',str('{:04.2f}'.format(accuracy_score*100))+'%')
print('------------------------------------------------')
print('Confusion Matrix:')
print(pd.DataFrame(confusion_matrix(y2_test, predicted)))
print('------------------------------------------------')
print('Classification Report:')
print(classification_report(y2_test, predicted))


# predict probabilities for MNB models:------------------------------------------
# #1----------------------------------
MNB_prob1 = MNB1.predict_proba(X1_test)

# #2----------------------------------
MNB_prob2 = MNB2.predict_proba(X2_test)


## TF-IDF: Term Frequency-Inverse Document Frequency ******************************************************************************

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()

text_count_2_1 = tfidf.fit_transform(nb1['wo_stopfreq_lem1'])
text_count_2_2 = tfidf.fit_transform(nb2['wo_stopfreq_lem2'])

#splitting the data in test and training
#from sklearn.model_selection() import train_test_split()
x1_train, x1_test, y1_train, y1_test = train_test_split(text_count_2_1, nb1['Sentiment'],test_size=0.20,random_state=30)
x2_train, x2_test, y2_train, y2_test = train_test_split(text_count_2_2, nb2['Sentiment'],test_size=0.20,random_state=30)


#fitting the model with MNB--------------------------
MNB1.fit(x1_train, y1_train)                                   #####1-------------
accuracy_score_mnb1 = metrics.accuracy_score(MNB1.predict(x1_test), y1_test)
print('accuracy_score_mnb1 = '+str('{:4.2f}'.format(accuracy_score_mnb1*100))+'%')

MNB2.fit(x2_train, y2_train)                                   #####2-------------
accuracy_score_mnb2 = metrics.accuracy_score(MNB2.predict(x2_test), y2_test)
print('accuracy_score_mnb2 = '+str('{:4.2f}'.format(accuracy_score_mnb2*100))+'%')





