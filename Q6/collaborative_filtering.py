from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict
from operator import itemgetter
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


movie_data_path = 'ratings_small.csv'
movie_data = pd.read_csv(movie_data_path)
movie_data.drop('timestamp', axis=1, inplace=True)
unique_movies = movie_data['movieId'].unique()
uniique_users = movie_data['userId'].unique()

print(movie_data.head())
print(f'SHAPE : {movie_data.shape}')
print(f'DESCRIPTION : \n{movie_data.describe()}')
print(f'INFO : \n{movie_data.info()}')
print(f'Total Unique Movies : {len(unique_movies)}')
print(f'Total Unique Users : {len(uniique_users)}')

active_users = movie_data['userId'].value_counts().head(1000).index
popular_movies = movie_data['movieId'].value_counts().head(1000).index

subset_data = movie_data[movie_data['userId'].isin(active_users) & movie_data['movieId'].isin(popular_movies)]
print(subset_data.head())
print(f'SHAPE : {subset_data.shape}')

user_movie_matrix = subset_data.pivot_table(index='userId', columns='movieId', values='rating')
user_movie_matrix.fillna(0, inplace=True)
print(user_movie_matrix.head())

user_similarity_matrix = cosine_similarity(user_movie_matrix)
np.fill_diagonal(user_similarity_matrix, 0)
user_similarity_df = pd.DataFrame(user_similarity_matrix, index=user_movie_matrix.index, columns=user_movie_matrix.index)
print(user_similarity_df.head())

print(user_similarity_df.index.max())

random_user = np.random.choice(user_movie_matrix.index)
print(random_user)

def get_top_similar_users(userId, user_similarity_df, threshold = 0, n =10):
    user_similarity_scores = user_similarity_df.iloc[userId, :]
    if threshold > 0:
        similar_users = user_similarity_scores[user_similarity_scores > threshold].sort_values(ascending=False). head(n)
    else:
        similar_users = user_similarity_scores.sort_values(ascending=False).head(n)
    return similar_users

similar_users = get_top_similar_users(random_user, user_similarity_df, threshold=0, n=10)
print(similar_users)
type(similar_users)
similar_users.index.to_list()

similar_user_ratings = movie_data[movie_data['userId'].isin(similar_users.index.to_list())]
print(similar_user_ratings.head())

# Candidate scoring:
# Add up ratings for each item, weighted by user similarity

scores=defaultdict(float)
for index, similar_user_rating in similar_user_ratings.iterrows():
    movie_id = similar_user_rating['movieId']
    user_rating = similar_user_rating['rating']
    user_id = similar_user_rating['userId']
    user_similarity_score = similar_users[int(user_id)]
    scores[int(movie_id)] += (user_rating / 5.0) * user_similarity_score

movie_data.loc[movie_data['userId'] == random_user].sort_values(by='rating', ascending=False).head(10)

watched = {}
for index, row in movie_data.loc[movie_data['userId'] == random_user].iterrows():
    watched[row['movieId']] = row['rating']



recommendations = {}
pos = 0

for movie, score in sorted(scores.items(), key=itemgetter(1), reverse=True):
    if movie not in watched:
        recommendations[movie] = score
        pos += 1
    if pos >= 10:
        break

for recommendation, score in recommendations.items():
    print(f"Movie : {recommendation} , Score : {score}")



# Load the data
data_path = "ratings_small.csv"
movie_data = pd.read_csv(data_path)

# Drop the timestamp column
movie_data.drop("timestamp", axis=1, inplace=True)

# Split the data into training and test sets
train_data, test_data = train_test_split(movie_data, test_size=0.2, random_state=42)

# Create a user-item matrix for training
train_user_item_matrix = train_data.pivot(index="userId", columns="movieId", values="rating").fillna(0)

# Calculate user similarity using cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
user_similarity_matrix = cosine_similarity(train_user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity_matrix, index=train_user_item_matrix.index, columns=train_user_item_matrix.index)

# Generate recommendations for users in the test set
def recommend_movies(user_id, user_similarity_df, train_user_item_matrix, top_n=10):
    if user_id not in user_similarity_df.index:
        return []  # No recommendations for users not in the training set
    similar_users = user_similarity_df.loc[user_id].sort_values(ascending=False).index
    scores = train_user_item_matrix.loc[similar_users].sum(axis=0)
    scores = scores.sort_values(ascending=False)
    recommended_movies = scores.index[:top_n]
    return recommended_movies

# Evaluate the recommendations
precision_list = []
recall_list = []
f1_list = []

for user_id in test_data["userId"].unique():
    # Ground truth: Movies the user interacted with in the test set
    ground_truth = test_data[test_data["userId"] == user_id]["movieId"].tolist()
   
    # Predicted: Top-N recommended movies
    if user_id in train_user_item_matrix.index:
        recommendations = recommend_movies(user_id, user_similarity_df, train_user_item_matrix, top_n=10)
    else:
        recommendations = []  # No recommendations for users not in the training set

    # Calculate precision, recall, and F1-score for this user
    if len(ground_truth) > 0:  # Avoid division by zero
        true_positives = len(set(recommendations) & set(ground_truth))
        precision = true_positives / len(recommendations) if len(recommendations) > 0 else 0
        recall = true_positives / len(ground_truth)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

# Calculate average metrics across all users
average_precision = sum(precision_list) / len(precision_list) if len(precision_list) > 0 else 0
average_recall = sum(recall_list) / len(recall_list) if len(recall_list) > 0 else 0
average_f1 = sum(f1_list) / len(f1_list) if len(f1_list) > 0 else 0

print(f"Average Precision: {average_precision:.4f}")
print(f"Average Recall: {average_recall:.4f}")
print(f"Average F1-score: {average_f1:.4f}")



# Load the data
data_path = "ratings_small.csv"
movie_data = pd.read_csv(data_path)

# Drop the timestamp column
movie_data.drop("timestamp", axis=1, inplace=True)

# Split the data into training and test sets
train_data, test_data = train_test_split(movie_data, test_size=0.2, random_state=42)

# Filter items to reduce matrix size
popular_items_item = train_data["movieId"].value_counts().head(2000).index
subset_data_item = train_data[train_data["movieId"].isin(popular_items_item)]
train_item_user_matrix = subset_data_item.pivot(index="movieId", columns="userId", values="rating").fillna(0)

# Calculate item similarity using cosine similarity
item_similarity_matrix = cosine_similarity(train_item_user_matrix)
item_similarity_df = pd.DataFrame(item_similarity_matrix, index=train_item_user_matrix.index, columns=train_item_user_matrix.index)

# Predict ratings for a user
def predict_ratings(user_id, train_item_user_matrix, item_similarity_df, top_n=10):
    if user_id not in train_item_user_matrix.columns:
        return []  # No recommendations for users not in the training set
   
    user_ratings = train_item_user_matrix[user_id]
   
    # Compute scores for items
    scores = item_similarity_df.dot(user_ratings) / np.abs(item_similarity_df).sum(axis=1)
    scores = pd.Series(scores, index=train_item_user_matrix.index)
   
    # Exclude items the user has already rated
    rated_items = user_ratings[user_ratings > 0].index
    scores = scores.drop(index=rated_items)
   
    # Recommend top-N items
    recommended_items = scores.sort_values(ascending=False).head(top_n).index.tolist()
    return recommended_items

# Evaluate the recommendations
precision_list = []
recall_list = []
f1_list = []

for user_id in test_data["userId"].unique():
    # Ground truth: Movies the user interacted with in the test set
    ground_truth = test_data[test_data["userId"] == user_id]["movieId"].tolist()
   
    # Predicted: Top-N recommended movies
    recommendations = predict_ratings(user_id, train_item_user_matrix, item_similarity_df, top_n=10)
   
    # Calculate precision, recall, and F1-score for this user
    if len(ground_truth) > 0:  # Avoid division by zero
        true_positives = len(set(recommendations) & set(ground_truth))
        precision = true_positives / len(recommendations) if len(recommendations) > 0 else 0
        recall = true_positives / len(ground_truth)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

# Calculate average metrics across all users
average_precision = np.mean(precision_list)
average_recall = np.mean(recall_list)
average_f1 = np.mean(f1_list)

print(f"Precision: {average_precision:.4f}")
print(f"Recall: {average_recall:.4f}")
print(f"F1-score: {average_f1:.4f}")