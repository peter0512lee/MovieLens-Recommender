from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib

# 初始化 FastAPI 應用
app = FastAPI()

# 添加 CORS 中間件配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500"],  # 允許的源
    allow_credentials=True,
    allow_methods=["*"],  # 允許的 HTTP 方法
    allow_headers=["*"],  # 允許的 HTTP 標頭
)

# 載入模型和數據
model = joblib.load('app/models/svd_model.pkl')  # 預先訓練好的模型
movies = pd.read_csv('data/movies.csv')          # 電影數據
ratings = pd.read_csv('data/cleaned_ratings.csv')  # 評分數據


@app.get("/")
def read_root():
    return {"message": "推薦系統 API 運行中"}


@app.get("/recommend/{user_id}")
def recommend_movies(user_id: int, n: int = 10):
    try:
        # 確認用戶是否存在
        if user_id not in ratings['userId'].unique():
            raise HTTPException(status_code=404, detail="用戶不存在")

        # 呼叫推薦函數
        recommendations = get_top_n_recommendations(model, user_id, movies, n)
        return recommendations.to_dict(orient='records')

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"發生錯誤: {str(e)}")


def get_top_n_recommendations(model, user_id, movie_df, n=10):
    user_rated_movies = ratings[ratings['userId'] == user_id]['movieId']
    all_movies = movie_df['movieId'].unique()
    movies_to_predict = [
        movie for movie in all_movies if movie not in user_rated_movies.values]

    # 預測未評分的電影
    predictions = [model.predict(user_id, movie_id)
                   for movie_id in movies_to_predict]
    predictions.sort(key=lambda x: x.est, reverse=True)

    # 回傳前 N 個推薦
    top_n = [(pred.iid, pred.est) for pred in predictions[:n]]
    return pd.DataFrame(top_n, columns=['movieId', 'predicted_rating']).merge(movie_df, on='movieId')
