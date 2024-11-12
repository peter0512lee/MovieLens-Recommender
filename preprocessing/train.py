import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split
import joblib

# 載入數據
ratings = pd.read_csv('data/cleaned_ratings.csv')
movies = pd.read_csv('data/movies.csv')

# 初始化 Reader 並載入數據到 Surprise 格式
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# 拆分訓練集和測試集
trainset, testset = train_test_split(data, test_size=0.2)

# 初始化 SVD 模型並訓練
model = SVD()
model.fit(trainset)

# 在測試集上進行預測和評估
predictions = model.test(testset)
rmse = accuracy.rmse(predictions)

# 保存模型
joblib.dump(model, 'app/models/svd_model.pkl')

# 定義推薦函數


def get_top_n_recommendations(model, user_id, movie_df, n=10):
    user_rated_movies = ratings[ratings['userId'] == user_id]['movieId']
    all_movies = movie_df['movieId'].unique()
    movies_to_predict = [
        movie for movie in all_movies if movie not in user_rated_movies.values]
    predictions = [model.predict(user_id, movie_id)
                   for movie_id in movies_to_predict]
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_n = [(pred.iid, pred.est) for pred in predictions[:n]]
    return pd.DataFrame(top_n, columns=['movieId', 'predicted_rating']).merge(movie_df, on='movieId')


# 測試推薦系統
recommendations = get_top_n_recommendations(model, 1, movies)
print("推薦結果:")
print(recommendations)
