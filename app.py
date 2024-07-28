import streamlit as st
import pandas as pd
import numpy as np
from pytube import YouTube
import pyperclip
import altair as alt
import joblib
import requests
import math

pipe_lr = joblib.load(open("model/text_emotion.pkl", "rb"))

emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", "neutral": "😐", "sad": "😔",
                       "sadness": "😔", "shame": "😳", "surprise": "😮"}

def fetch_comments(video_url):
    full_comments = []

    try:
        # Get video ID from the URL
        video_id = YouTube(video_url).video_id

        # Fetch comments using YouTube API
        url = f"https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId={video_id}&key=AIzaSyCIqUvMIWesFd3jKdKS_VxrUZqUAvbQwrU"
        response = requests.get(url).json()

        # Extract comments
        if 'items' in response:
            comments = [item['snippet']['topLevelComment']['snippet']['textDisplay'] for item in response['items']]
            full_comments.extend(comments)
        else:
            st.error("Error: 'items' key not found in the response.")
    except Exception as e:
        st.error(f"Error: {e}")
        # Handle the error, such as setting data to an empty list or logging the issue
        full_comments = []

    return full_comments

def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]


def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results


def main():
    st.title("Text Emotion Detection")
    st.subheader("Detect Emotions In Text")

    with st.form(key='my_form1'):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label='Submit')

    if submit_text:
        col1, col2 = st.columns(2)

        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        with col1:
            st.success("Original Text")
            st.write(raw_text)

            st.success("Prediction")
            emoji_icon = emotions_emoji_dict[prediction]
            st.write("{}:{}".format(prediction, emoji_icon))
            st.write("Confidence: {}%".format(round(np.max(probability*100),2)))

        with col2:
            st.success("Prediction Probability")
            #st.write(probability)
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            #st.write(proba_df.T)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]

            fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
            st.altair_chart(fig, use_container_width=True)
    st.title("YouTube Comment Extractor")
    # User input for YouTube video URL
    video_url = st.text_input("Enter the YouTube video URL:")

    if st.button("Extract Comments"):
        st.caption("Copy and Paste any of the comment in the above emotion detection Box to predict.")
        comments = fetch_comments(video_url)
        st.write("### Comments Extracted:")
        for index, comment in enumerate(comments):
            st.write(f"{index + 1}. {comment}")       

if __name__ == '__main__':
    main()
