from youtube_comment_scraper_python import*
import pandas as pd
import time
url = input("Enter the video url")
file = input("Enter the output file name = ")
youtube.open(url)
fullcomments= []
for i in range(0,1):
    result=youtube.video_comments()
    data=result['body']
    fullcomments.extend(data)
dataframe=pd. DataFrame (fullcomments)
print (dataframe)
time. sleep(5)
dataframe.to_csv(file)