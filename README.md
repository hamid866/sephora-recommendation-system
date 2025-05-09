# Sephora Recommendation System 

1. **Data Preparation**  
2. **Collaborative Filtering**  
3. **SVD-based Model-Based CF & Hyperparameter Tuning**  
4. **Content-Based & Hybrid Recommendation**  
5. **Evaluation**

---

Data Preparation:

Loaded product_info.csv and all segmented reviews_*.csv into pandas.

Renamed columns to userID, itemID, rating, dropped any rows missing these.

Combined into a single reviews DataFrame (1.1 M rows, 2 351 items, 578 653 users) and prepared a Surprise‐compatible Dataset.

Collaborative Filtering:

User-based KNN (cosine similarity) and Item-based KNN with Surprise’s KNNBasic.

Evaluated both on an 80/20 train/test split using RMSE & MAE.

Model-based CF (SVD) & Hyperparameter Tuning:

Applied Surprise’s SVD().

Ran a GridSearchCV over

n_factors ∈ {50, 100, 150}

lr_all ∈ {0.002, 0.005}

reg_all ∈ {0.02, 0.1}

Selected best combo, retrained on trainset, and reported final RMSE/MAE on the hold-out test set.

Plotted mean CV-RMSE vs. n_factors (and vs. reg_all/lr_all).

Content-Based & Hybrid Recommendations:

TF-IDF CB: concatenated product_name + brand_name + primary_category, vectorized with TfidfVectorizer, built per-user profiles by rating-weighted average, and recommended by cosine similarity.

Text-Embedding CB: used your 64-dim TruncatedSVD product embeddings, averaged per-user (weighted by ratings), and recommended by cosine similarity.

Three-Signal Hybrid: blended SVD-predicted ratings, TF-IDF similarities (scaled), and text-embedding similarities with tunable weights (α, β).

Evaluation Metrics:

Rating Accuracy: RMSE & MAE for User-KNN, Item-KNN, baseline & tuned SVD.

Ranking Quality: Precision@10 & Recall@10 for all four methods, estimated via a 100-user random sample.

User-Fingerprint Regression: aggregated per-user features (rating_avg, rating_count), trained a RandomForest to predict rating_avg, swept n_estimators, and plotted Test MSE vs. n_estimators.

