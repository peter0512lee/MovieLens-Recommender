import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 載入數據
ratings = pd.read_csv('data/ratings.csv')
movies = pd.read_csv('data/movies.csv')

# 檢查並移除遺失值
ratings = ratings.dropna()
movies = movies.dropna()

# 合併數據集
merged_data = pd.merge(ratings, movies, on='movieId')

# 計算每部電影的平均評分和評分次數
movie_stats = merged_data.groupby('title').agg({'rating': ['mean', 'count']})
movie_stats.columns = ['average_rating', 'rating_count']
movie_stats = movie_stats.reset_index()

# 過濾評分次數少於 50 次的電影
popular_movies = movie_stats[movie_stats['rating_count'] > 50]

# 資料標準化到 0-1 之間
scaler = MinMaxScaler()
popular_movies['scaled_rating'] = scaler.fit_transform(
    popular_movies[['average_rating']])

# 保存處理後的資料
merged_data.to_csv('data/cleaned_ratings.csv', index=False)
popular_movies.to_csv('data/popular_movies.csv', index=False)

print("資料前處理完成並保存。")
