import os
import googleapiclient.discovery
import pandas as pd
#https://youtu.be/oTnhx_N7nro?si=XeJj8TzMOQX4P2GP (video link)


API_KEY = os.environ.get("API_KEY")

video_id = "oTnhx_N7nro"

# API call
youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=API_KEY)

# Function to scrape comments
def scrape_comments(video_id):
    comments = []

    # Get video comments
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100,  # Adjust maxResults as needed
        textFormat="plainText"
    )

    while request:
        response = request.execute()

        # Collect comments
        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)

        # Check if there are more comments
        request = youtube.commentThreads().list_next(request, response)

    return comments


video_comments = scrape_comments(video_id)

# Pandas DataFrame with comments
df = pd.DataFrame({"Comment": video_comments})

# Export comments extracted
df.to_csv("youtube_comments.csv", index=False)

