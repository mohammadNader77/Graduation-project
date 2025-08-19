import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import warnings
import json

warnings.filterwarnings("ignore")

# Load datasets
youtubeUS = pd.read_csv('C:\\Users\\user\\Desktop\\Testtt\\data\\combined_dataset (1).csv', encoding='latin-1')
CategoryCSV = pd.read_csv('C:\\Users\\user\\Desktop\\Testtt\\data\\test.csv')

category_mapping = {
    "Film & Animation": 1,
    "Autos & Vehicles": 2,
    "Music": 10,
    "Pets & Animals": 15,
    "Sports": 17,
    "Short Movies": 18,
    "Travel & Events": 19,
    "Gaming": 20,
    "Videoblogging": 21,
    "People & Blogs": 22,
    "Comedy": 23,
    "Entertainment": 24,
    "News & Politics": 25,
    "Howto & Style": 26,
    "Education": 27,
    "Science & Technology": 28,
    "Nonprofits & Activism": 29,
    "Movies": 30,
    "Anime/Animation": 31,
    "Action/Adventure": 32,
    "Classics": 33,
    "Documentary": 35,
    "Drama": 36,
    "Family": 37,
    "Foreign": 38,
    "Horror": 39,
    "Sci-Fi/Fantasy": 40,
    "Thriller": 41,
    "Shorts": 42,
    "Shows": 43,
    "Trailers": 44
}

print("Starting data preprocessing...")

# Prepare data
Trend_Video_US = youtubeUS.drop(['comments_disabled', 'ratings_disabled', 'description', 'tags', 'thumbnail_link', 'video_error_or_removed'], axis=1)
Trend_Video_US = pd.merge(Trend_Video_US, CategoryCSV.rename(columns={'id': 'category_id', 'title': 'category'}), how="left", on="category_id")

# Convert dates
Trend_Video_US['trending_date'] = pd.to_datetime(Trend_Video_US['trending_date'], format='%y.%d.%m', errors='coerce')
Trend_Video_US['publish_date'] = pd.to_datetime(Trend_Video_US['publish_time'], errors='coerce')
Trend_Video_US['trending_date'] = Trend_Video_US['trending_date'].dt.tz_localize(None)
Trend_Video_US['publish_date'] = Trend_Video_US['publish_date'].dt.tz_localize(None)

# Remove rows with invalid dates
Trend_Video_US = Trend_Video_US.dropna(subset=['trending_date', 'publish_date'])

# Calculate days to trend
Trend_Video_US['Days_to_Trend'] = (Trend_Video_US['trending_date'] - Trend_Video_US['publish_date']).dt.days

# Remove impossible values (negative days or extremely long periods)
Trend_Video_US = Trend_Video_US[(Trend_Video_US['Days_to_Trend'] >= 0) & (Trend_Video_US['Days_to_Trend'] <= 365)]

# CORRECTED: Define target variable - 1 for fast trending (≤7 days), 0 for slow/non-trending
Trend_Video_US['will_trend_fast'] = np.where(Trend_Video_US['Days_to_Trend'] <= 7, 1, 0)

# Convert numeric columns
numeric_columns = ['likes', 'dislikes', 'views', 'comment_count']
for col in numeric_columns:
    Trend_Video_US[col] = pd.to_numeric(Trend_Video_US[col], errors='coerce')

# Remove rows with missing numeric data
Trend_Video_US = Trend_Video_US.dropna(subset=numeric_columns)

# Map categories
Trend_Video_US['category'] = Trend_Video_US['category'].map(category_mapping)
Trend_Video_US = Trend_Video_US.dropna(subset=['category'])
Trend_Video_US['category'] = Trend_Video_US['category'].astype(int)

# Create improved features
Trend_Video_US['publish_hour'] = Trend_Video_US['publish_date'].dt.hour
Trend_Video_US['publish_day'] = Trend_Video_US['publish_date'].dt.day
Trend_Video_US['publish_month'] = Trend_Video_US['publish_date'].dt.month
Trend_Video_US['publish_weekday'] = Trend_Video_US['publish_date'].dt.weekday
Trend_Video_US['publish_time_numeric'] = Trend_Video_US['publish_date'].astype(np.int64) // 10**9

# Better engagement features
Trend_Video_US['like_rate'] = Trend_Video_US['likes'] / (Trend_Video_US['views'] + 1)
Trend_Video_US['dislike_rate'] = Trend_Video_US['dislikes'] / (Trend_Video_US['views'] + 1)
Trend_Video_US['comment_rate'] = Trend_Video_US['comment_count'] / (Trend_Video_US['views'] + 1)
Trend_Video_US['engagement_rate'] = (Trend_Video_US['likes'] + Trend_Video_US['dislikes'] + Trend_Video_US['comment_count']) / (Trend_Video_US['views'] + 1)

# Title features
Trend_Video_US['title_length'] = Trend_Video_US['title'].str.len()
Trend_Video_US['title_word_count'] = Trend_Video_US['title'].str.split().str.len()
Trend_Video_US['title_upper_ratio'] = Trend_Video_US['title'].str.count(r'[A-Z]') / (Trend_Video_US['title_length'] + 1)

# Time-based features
Trend_Video_US['is_weekend'] = (Trend_Video_US['publish_weekday'] >= 5).astype(int)
Trend_Video_US['is_prime_time'] = ((Trend_Video_US['publish_hour'] >= 18) & (Trend_Video_US['publish_hour'] <= 22)).astype(int)

# Log transform for skewed features
log_features = ['views', 'likes', 'dislikes', 'comment_count']
for col in log_features:
    Trend_Video_US[f'log_{col}'] = np.log1p(Trend_Video_US[col])

# Encode categorical variables
title_encoder = LabelEncoder()
category_encoder = LabelEncoder()

# Handle unknown titles by assigning a special value
unique_titles = Trend_Video_US['title'].unique()
title_encoder.fit(list(unique_titles) + ['UNKNOWN_TITLE'])
Trend_Video_US['title_encoded'] = title_encoder.transform(Trend_Video_US['title'])

# Handle categories
unique_categories = Trend_Video_US['category'].unique()
category_encoder.fit(list(unique_categories) + [-1])  # -1 for unknown
Trend_Video_US['category_encoded'] = category_encoder.transform(Trend_Video_US['category'])

# Feature selection
feature_columns = [
    'log_views', 'log_likes', 'log_dislikes', 'log_comment_count',
    'like_rate', 'dislike_rate', 'comment_rate', 'engagement_rate',
    'title_length', 'title_word_count', 'title_upper_ratio',
    'category_encoded', 'publish_hour', 'publish_weekday',
    'publish_month', 'is_weekend', 'is_prime_time'
]

X = Trend_Video_US[feature_columns]
Y = Trend_Video_US['will_trend_fast']

# Remove any remaining NaN values
df_cleaned = pd.concat([X, Y], axis=1).dropna()
X = df_cleaned[feature_columns]
Y = df_cleaned['will_trend_fast']

print("Class distribution:")
print(Y.value_counts())
print(f"Percentage of fast-trending videos: {Y.mean()*100:.2f}%")

# Feature selection - keep more relevant features
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, Y)
selected_features = [feature_columns[i] for i in selector.get_support(indices=True)]
print("Selected features:", selected_features)

# Feature importance scores
feature_scores = selector.scores_
feature_importance = list(zip(feature_columns, feature_scores))
feature_importance.sort(key=lambda x: x[1], reverse=True)
print("\nFeature importance (top 10):")
for feat, score in feature_importance[:10]:
    print(f"{feat}: {score:.2f}")

# Stratified split
X_train, X_test, Y_train, Y_test = train_test_split(X_selected, Y, test_size=0.2, random_state=42, stratify=Y)

# Scale features for better performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Improved RandomForest with better parameters
random_forest = RandomForestClassifier(
    random_state=42,
    n_estimators=100,  # More trees for better performance
    max_depth=10,      # Deeper trees
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    n_jobs=-1,
    bootstrap=True
)

print("\nTraining model...")
random_forest.fit(X_train_scaled, Y_train)

# Cross-validation
cv_scores = cross_val_score(random_forest, X_train_scaled, Y_train, cv=5, scoring='roc_auc')
print(f"Cross-validation ROC AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Predictions
y_pred_rf = random_forest.predict(X_test_scaled)
y_prob_rf = random_forest.predict_proba(X_test_scaled)[:, 1]

# Evaluation
print(f"\nModel Performance:")
print(f"Accuracy: {accuracy_score(Y_test, y_pred_rf):.4f}")
print(f"ROC AUC Score: {roc_auc_score(Y_test, y_prob_rf):.4f}")

print("\nClassification Report:")
print(classification_report(Y_test, y_pred_rf))

print("\nConfusion Matrix:")
cm = confusion_matrix(Y_test, y_pred_rf)
print(cm)

# Feature importance from the model
feature_importances = random_forest.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': selected_features,
    'importance': feature_importances
}).sort_values('importance', ascending=False)

print("\nTop 10 most important features:")
print(feature_importance_df.head(10))

# Save all necessary components
print("\nSaving model and encoders...")
joblib.dump(random_forest, 'random_forest_model.pkl')
joblib.dump(title_encoder, 'title_encoder.pkl')
joblib.dump(category_encoder, 'category_encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Save feature information
model_info = {
    'selected_features': selected_features,
    'feature_columns': feature_columns,
    'category_mapping': category_mapping
}

with open('model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print("Model training completed successfully!")
print(f"Model saved with {len(selected_features)} features")
print("Files saved: random_forest_model.pkl, title_encoder.pkl, category_encoder.pkl, scaler.pkl, model_info.json")