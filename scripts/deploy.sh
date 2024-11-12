#!/bin/bash

# 構建 Docker 映像
docker build -t recommendation-api .

# 標記並推送至 GCP Container Registry
docker tag recommendation-api gcr.io/[YOUR_PROJECT_ID]/recommendation-api
docker push gcr.io/[YOUR_PROJECT_ID]/recommendation-api

# 部署至 Cloud Run
gcloud run deploy recommendation-api \
    --image gcr.io/[YOUR_PROJECT_ID]/recommendation-api \
    --platform managed \
    --region [YOUR_REGION] \
    --allow-unauthenticated