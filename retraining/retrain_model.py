import pandas as pd
from surprise import SVD, Dataset, Reader
import joblib

# 載入數據
ratings = pd.read_csv('data/ratings.csv')
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# 訓練新模型
model = SVD()
trainset = data.build_full_trainset()
model.fit(trainset)

# 保存模型
joblib.dump(model, 'app/models/svd_model.pkl')
print("Model retrained and saved successfully.")
