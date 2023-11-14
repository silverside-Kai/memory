import math
import pymongo
from config import mongo_string, pgvector_string
import time
from datetime import datetime
from sqlalchemy import create_engine
import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np


postgresql_engine = create_engine(
    pgvector_string
)
mongo_collection = pymongo.MongoClient(mongo_string)['mvp']['source']


def calculate_blog_score(blog_creation_time, source_importance, current_time, max_param, alpha, source_weight):
    time_difference = (current_time - blog_creation_time) / 60 / 60 / 24
    score = math.exp(-((max_param-source_importance) ** source_weight) * alpha * time_difference)
    return score


query = {"prior_importance": {"$exists": True}}
data = mongo_collection.find(query)

max_param = 10.05
alpha = 0.01
beta = 1.2

current_time = int(time.time())
current_date = datetime.now().strftime('%Y-%m-%d')

ids = []
titles = []
links = []
sources = []
types = []
scores = []
timestamps = []
current_dates = []

for index, document in enumerate(data):
    timestamp_str = document['created_time']
    source_importance = document['prior_importance']
    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00')).timestamp()
    timestamp = int(timestamp)
    score = calculate_blog_score(timestamp, int(source_importance), current_time, max_param, alpha, beta)
    mongo_collection.update_one(
        {"_id": document['_id']},
        {"$set": {"score": score}}
    )
    id_str = str(document['_id'])
    title_str = document['title']
    link_str = document['link']
    source_str = document['source']
    type_str = document['type']
    score_str = str(score)

        # Append data to lists
    ids.append(id_str)
    titles.append(title_str)
    links.append(link_str)
    sources.append(source_str)
    types.append(type_str)
    scores.append(score_str)
    timestamps.append(timestamp_str)
    current_dates.append(current_date)

# Create a DataFrame from the lists
df = pd.DataFrame({
    'doc_id': ids,
    'title': titles,
    'link': links,
    'source': sources,
    'content_type': types,
    'score': scores,
    'timestamp': timestamps,
    'current_date': current_dates
})
df.to_sql(name='ranking_scores', schema='memory', if_exists='append', index=False,
          con=postgresql_engine, method='multi')

# Illustrate the decay function
# source_importance_values = [2, 4, 6, 8, 10]
# colors = {2: 'red', 4: 'blue', 6: 'green', 8: 'orange', 10: 'purple'}
# # Create an array of x-values, where each value represents a different number of days (1 to 30)
# days = np.arange(0, 30)  # 0 to 30 days

# # Loop through source_importance values and create scatter plots
# for source_importance in source_importance_values:
#     x_values = []  # Create an empty list for x-values
#     scores = []

#     for day in days:
#         timestamp = current_time - (day * 60 * 60 * 24)
#         x_values.append((current_time - timestamp) / (60 * 60 * 24))  # Calculate x-values here
#         score = calculate_blog_score(timestamp, source_importance, current_time, max_param, alpha, beta)
#         scores.append(score)

#     plt.scatter(x_values, scores, label=f'Source Importance {source_importance}', color=colors[source_importance])

# # Add labels and legend
# plt.xlabel('Days Ago')
# plt.ylabel('Score')
# plt.legend()

# # Display the plot
# plt.show()
