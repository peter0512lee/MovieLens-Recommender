import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split
import joblib
import numpy as np
import random

# 設定隨機種子以確保結果可重現
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 載入數據
ratings = pd.read_csv('data/cleaned_ratings.csv')
movies = pd.read_csv('data/movies.csv')

# 初始化 Reader 並載入數據到 Surprise 格式
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# 拆分訓練集和測試集，加入隨機種子
trainset, testset = train_test_split(
    data, test_size=0.2, random_state=RANDOM_SEED)

# 初始化 SVD 模型並訓練，加入隨機種子
model = SVD(random_state=RANDOM_SEED)
model.fit(trainset)

# 在測試集上進行預測和評估
predictions = model.test(testset)
rmse = accuracy.rmse(predictions)

# 保存模型
joblib.dump(model, 'app/models/svd_model.pkl')

# 定義推薦函數


def get_top_n_recommendations(model, user_id, movie_df, n=10):
    """
    為指定用戶生成電影推薦清單

    參數:
        model: 訓練好的 SVD 模型
        user_id: 目標用戶 ID
        movie_df: 包含電影資訊的 DataFrame
        n: 要推薦的電影數量 (默認為10部)

    返回:
        包含推薦電影資訊的 DataFrame
    """
    # 獲取用戶已經評分過的電影列表
    user_rated_movies = ratings[ratings['userId'] == user_id]['movieId']

    # 獲取所有電影的 ID
    all_movies = movie_df['movieId'].unique()

    # 過濾出用戶未評分的電影
    movies_to_predict = [
        movie for movie in all_movies if movie not in user_rated_movies.values]

    # 對每部未評分的電影進行評分預測
    predictions = [model.predict(user_id, movie_id)
                   for movie_id in movies_to_predict]

    # 根據預測評分從高到低排序
    predictions.sort(key=lambda x: x.est, reverse=True)

    # 選取前 n 個最高評分的電影
    top_n = [(pred.iid, pred.est) for pred in predictions[:n]]

    # 將結果轉換為 DataFrame 並與電影資訊合併
    return pd.DataFrame(top_n, columns=['movieId', 'predicted_rating']).merge(movie_df, on='movieId')


# 測試推薦系統
recommendations = get_top_n_recommendations(model, 1, movies)
print("推薦結果:")
print(recommendations)
