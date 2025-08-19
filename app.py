from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import json
import requests
import re
from datetime import datetime, timedelta
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Load model and preprocessing components
try:
    model = joblib.load('random_forest_model.pkl')
    title_encoder = joblib.load('title_encoder.pkl')
    category_encoder = joblib.load('category_encoder.pkl')
    scaler = joblib.load('scaler.pkl')
    
    with open('model_info.json', 'r') as f:
        model_info = json.load(f)
    
    selected_features = model_info['selected_features']
    feature_columns = model_info['feature_columns']
    category_mapping = model_info['category_mapping']
    # Create reverse mapping for categories (id to name)
    id_to_category = {v: k for k, v in category_mapping.items()}
    
    print("Model and preprocessing components loaded successfully!")
    print(f"Selected features: {selected_features}")
    
except Exception as e:
    print(f"Error loading model components: {e}")
    raise

# YouTube API configuration
YOUTUBE_API_KEY = 'AIzaSyCCw9P0xNcMx5KKWSpzxVodol4I4Pd82Yo'  # Replace with your API key

def get_category_name_by_id(category_id):
    """Get category name from ID using the reverse mapping"""
    return id_to_category.get(category_id, "Unknown")

def extract_video_id(url):
    """Extract video ID from various YouTube URL formats"""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11})',
        r'youtu\.be\/([0-9A-Za-z_-]{11})',
        r'embed\/([0-9A-Za-z_-]{11})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_video_info(video_id):
    """Fetch video information from YouTube API"""
    try:
        api_url = f'https://www.googleapis.com/youtube/v3/videos?part=snippet,statistics&id={video_id}&key={YOUTUBE_API_KEY}'
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if 'items' not in data or not data['items']:
            return None

        video = data['items'][0]
        snippet = video['snippet']
        stats = video['statistics']

        return {
            'title': snippet.get('title', ''),
            'views': int(stats.get('viewCount', 0)),
            'likes': int(stats.get('likeCount', 0)),
            'dislikes': int(stats.get('dislikeCount', 0)),  # Note: may be 0 due to YouTube changes
            'comment_count': int(stats.get('commentCount', 0)),
            'publish_date': pd.to_datetime(snippet.get('publishedAt')),
            'category_id': int(snippet.get('categoryId', -1)),
            'channel_title': snippet.get('channelTitle', ''),
            'description': snippet.get('description', '')[:200] + '...' if len(snippet.get('description', '')) > 200 else snippet.get('description', '')
        }
    except Exception as e:
        logging.error(f"Error fetching video info: {e}")
        return None

def create_features(title, views, likes, dislikes, comment_count, category_id, publish_date):
    """Create feature vector for prediction"""
    try:
        now = pd.Timestamp.now().tz_localize(None)
        
        # Basic engagement features
        like_rate = likes / (views + 1)
        dislike_rate = dislikes / (views + 1)
        comment_rate = comment_count / (views + 1)
        engagement_rate = (likes + dislikes + comment_count) / (views + 1)
        
        # Title features
        title_length = len(title)
        title_word_count = len(title.split())
        title_upper_ratio = sum(1 for c in title if c.isupper()) / (len(title) + 1)
        
        # Time features
        publish_hour = publish_date.hour
        publish_weekday = publish_date.weekday()
        publish_month = publish_date.month
        is_weekend = 1 if publish_weekday >= 5 else 0
        is_prime_time = 1 if 18 <= publish_hour <= 22 else 0
        
        # Log features
        log_views = np.log1p(views)
        log_likes = np.log1p(likes)
        log_dislikes = np.log1p(dislikes)
        log_comment_count = np.log1p(comment_count)
        
        # Encode categorical features
        try:
            title_encoded = title_encoder.transform([title])[0]
        except:
            # Handle unknown titles
            title_encoded = title_encoder.transform(['UNKNOWN_TITLE'])[0]
        
        try:
            category_encoded = category_encoder.transform([category_id])[0]
        except:
            # Handle unknown categories
            category_encoded = category_encoder.transform([-1])[0]
        
        # Create feature dictionary
        features = {
            'log_views': log_views,
            'log_likes': log_likes,
            'log_dislikes': log_dislikes,
            'log_comment_count': log_comment_count,
            'like_rate': like_rate,
            'dislike_rate': dislike_rate,
            'comment_rate': comment_rate,
            'engagement_rate': engagement_rate,
            'title_length': title_length,
            'title_word_count': title_word_count,
            'title_upper_ratio': title_upper_ratio,
            'category_encoded': category_encoded,
            'publish_hour': publish_hour,
            'publish_weekday': publish_weekday,
            'publish_month': publish_month,
            'is_weekend': is_weekend,
            'is_prime_time': is_prime_time
        }
        
        # Extract only selected features in correct order
        feature_vector = np.array([[features[feat] for feat in selected_features]])
        
        return feature_vector
        
    except Exception as e:
        logging.error(f"Error creating features: {e}")
        raise

@app.route('/')
def home():
    return render_template('index.html', categories=category_mapping)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        method = request.form.get('method')
        result = None
        video_info = None
        confidence = None
        
        now = pd.Timestamp.now().tz_localize(None)
        
        if method == 'manual':
            # Manual input prediction
            views = int(request.form['views'])
            likes = int(request.form['likes'])
            dislikes = int(request.form.get('dislikes', 0))
            comment_count = int(request.form['comment_count'])
            title = request.form['title'].strip()
            # Get category ID from form and convert to int
            category_id = int(request.form['category_encoded'])
            category_name = get_category_name_by_id(category_id)
            publish_date_str = request.form['publish_date']
            
            publish_date = pd.to_datetime(publish_date_str).tz_localize(None)
            
            # Validation
            if publish_date > now:
                return render_template('index.html', 
                                     result="⚠️ Publish date cannot be in the future.", 
                                     categories=category_mapping)
            
            # Check if video is too old for meaningful prediction
            days_since_publish = (now - publish_date).days
            if days_since_publish > 30:
                return render_template('index.html', 
                                     result="⚠️ Video is too old for trending prediction (>30 days).", 
                                     categories=category_mapping)
            
            video_info = {
                'title': title,
                'views': views,
                'likes': likes,
                'dislikes': dislikes,
                'comment_count': comment_count,
                'category': category_name,
                'category_id': category_id,
                'publish_date': publish_date.strftime('%Y-%m-%d %H:%M'),
                'days_since_publish': days_since_publish,
                'method': 'manual'
            }
            
        elif method == 'url':
            # URL-based prediction
            url = request.form['video_url'].strip()
            video_id = extract_video_id(url)
            
            if not video_id:
                return render_template('index.html', 
                                     result="⚠️ Invalid YouTube URL format.", 
                                     categories=category_mapping)
            
            video_data = get_video_info(video_id)
            if not video_data:
                return render_template('index.html', 
                                     result="⚠️ Unable to retrieve video data. Check the URL and API key.", 
                                     categories=category_mapping)
            
            title = video_data['title']
            views = video_data['views']
            likes = video_data['likes']
            dislikes = video_data['dislikes']
            comment_count = video_data['comment_count']
            publish_date = video_data['publish_date'].tz_localize(None)
            category_id = video_data['category_id']
            category_name = get_category_name_by_id(category_id)
            
            days_since_publish = (now - publish_date).days
            
            # Validation
            if days_since_publish > 30:
                return render_template('index.html', 
                                     result="⚠️ Video is too old for trending prediction (>30 days).", 
                                     categories=category_mapping)
            
            video_info = {
                'title': title,
                'views': views,
                'likes': likes,
                'dislikes': dislikes,
                'comment_count': comment_count,
                'category': category_name,
                'category_id': category_id,
                'publish_date': publish_date.strftime('%Y-%m-%d %H:%M'),
                'days_since_publish': days_since_publish,
                'channel_title': video_data.get('channel_title', ''),
                'description': video_data.get('description', ''),
                'url': url,
                'method': 'url'
            }
        
        else:
            return render_template('index.html', 
                                 result="⚠️ Invalid prediction method.", 
                                 categories=category_mapping)
        
        # Create features and make prediction
        features = create_features(title, views, likes, dislikes, comment_count, 
                                 category_id, publish_date)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0]
        
        # Get confidence score
        confidence = max(prediction_proba) * 100
        
        # Interpret prediction (0 = will trend fast, 1 = won't trend fast)
        if prediction == 0:
            result = f"✅ This video is likely to trend fast (within 30 days)! Confidence: {confidence:.1f}%"
        else:
            result = f"❌ This video is unlikely to trend fast. Confidence: {confidence:.1f}%"
        
        # Add additional insights
        engagement_rate = (likes + dislikes + comment_count) / (views + 1)
        like_ratio = likes / (views + 1) if views > 0 else 0
        
        insights = []
        if engagement_rate > 0.05:
            insights.append("🚀 High engagement rate detected!")
        elif engagement_rate < 0.01:
            insights.append("⚠️ Low engagement rate - needs more interactions")
            
        if like_ratio > 0.02:
            insights.append("👍 Strong like ratio!")
        elif like_ratio < 0.005:
            insights.append("⚠️ Low like ratio - consider improving content quality")
            
        if len(title) > 60:
            insights.append("ℹ️ Title might be too long for optimal performance")
        elif len(title) < 20:
            insights.append("ℹ️ Title might be too short to attract viewers")
            
        if video_info['days_since_publish'] == 0:
            insights.append("🆕 Video is very fresh - predictions may be less reliable")
            
        # Add category-specific insights
        category_insights = {
            1: "Gaming content often trends with high engagement",
            10: "Music videos typically need strong initial views",
            17: "Sports content benefits from timely events",
            20: "Gaming tutorials need clear explanations",
            22: "People & Blogs rely on personal connection",
            23: "Comedy benefits from shareable moments",
            24: "Entertainment videos need high engagement",
            25: "News/educational content requires timely topics",
            26: "How-to/style videos need practical value",
            27: "Education thrives on clear explanations",
            28: "Science/Tech needs accurate information"
        }
        
        # Add category-specific insight if available
        if category_id in category_insights:
            insights.append(f"💡 Category Insight: {category_insights[category_id]}")
        
        # Add additional category-based recommendations
        if category_id == 1 and engagement_rate < 0.03:  # Gaming
            insights.append("⚠️ Gaming content typically needs higher engagement to trend")
        elif category_id == 10 and views < 10000:  # Music
            insights.append("⚠️ Music videos usually need strong initial views to trend")
        # elif category_id == 24 and like_rate < 0.015:  # Entertainment
        #     insights.append("⚠️ Entertainment videos require higher like ratios to trend")
        elif category_id == 25 and days_since_publish > 1:  # News
            insights.append("⚠️ News content needs to trend quickly to be relevant")
        elif category_id == 27 and comment_count < 50:  # Education
            insights.append("⚠️ Educational content benefits from discussion in comments")
        
        return render_template('index.html', 
                             result=result, 
                             video_info=video_info,
                             insights=insights,
                             categories=category_mapping)
        
    except ValueError as ve:
        return render_template('index.html', 
                             result=f"⚠️ Input Error: {ve}", 
                             categories=category_mapping)
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return render_template('index.html', 
                             result="⚠️ An error occurred during prediction. Please try again.", 
                             categories=category_mapping)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)