from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Last.fm API key
lastfm_api_key = "cba62bbe77c9d608d23604ac0c5eb6e9"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    artist_name = data.get("artist")

    if not artist_name:
        return jsonify({'recommendations': {'error': 'No artist provided'}})

    recommendations, plot_path = get_recommendations(artist_name)
    return jsonify({'recommendations': recommendations, 'plot_path': plot_path})

def get_recommendations(artist_name):
    artist_name = artist_name.strip().lower()
    url = f"http://ws.audioscrobbler.com/2.0/?method=artist.getsimilar&artist={artist_name}&api_key={lastfm_api_key}&format=json"
    response = requests.get(url)
    data = response.json()

    if 'error' in data or 'similarartists' not in data:
        return {"error": "Error retrieving artist info. Please try again."}, None

    artists = data['similarartists']['artist'][:10]  # limit to top 10

    # Extract features for clustering
    artist_data = []
    for artist in artists:
        match_score = float(artist.get('match', 0))
        listeners = fetch_listener_count(artist['name'])  # optional
        artist_data.append({
            'name': artist['name'],
            'url': artist['url'],
            'match': match_score,
            'listeners': listeners,
            'image': next((img['#text'] for img in artist['image'] if img['#text']), 'https://via.placeholder.com/100')
        })

    df = pd.DataFrame(artist_data)

    # Normalize features
    feature_df = df[['match', 'listeners']].fillna(0)

    # Find the best k dynamically (Elbow method approximation)
    inertias = []
    K = range(1, min(6, len(feature_df)))  # Test up to 5 clusters (or fewer if data is small)

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(feature_df)
        inertias.append(kmeans.inertia_)

    # Detect the elbow: pick the smallest k where inertia reduction slows down
    k_best = 1
    for i in range(1, len(inertias)):
        if inertias[i-1] - inertias[i] < 0.1 * inertias[i-1]:  # Less than 10% drop
            k_best = i
            break
    else:
        k_best = len(inertias)

    # Fit KMeans using the chosen k
    kmeans = KMeans(n_clusters=k_best, random_state=0)
    kmeans.fit(feature_df)
    df['cluster'] = kmeans.labels_

    # Plot the clusters and return the plot path
    plot_path = plot_clusters(df)

    # Organize results by cluster
    clustered_results = {}
    for cluster in sorted(df['cluster'].unique()):
        clustered_results[f'Cluster {cluster}'] = df[df['cluster'] == cluster][['name', 'url', 'image']].to_dict(orient='records')

    return clustered_results, plot_path

def fetch_listener_count(artist_name):
    url = f"http://ws.audioscrobbler.com/2.0/?method=artist.getinfo&artist={artist_name}&api_key={lastfm_api_key}&format=json"
    response = requests.get(url)
    data = response.json()
    try:
        return int(data['artist']['stats']['listeners'])
    except:
        return 0

def plot_clusters(df):
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(df['match'], df['listeners'], c=df['cluster'], cmap='viridis', s=100)
    plt.title('Artist Clusters (Match vs Listeners)')
    plt.xlabel('Match Score')
    plt.ylabel('Listeners')
    plt.colorbar(scatter)
    plt.grid(True)
    
    # Ensure this is correctly saving to the static folder
    plot_path = os.path.join('static', 'cluster_plot.png')
    plt.savefig(plot_path)
    plt.close()  # To avoid memory issues
    
    return plot_path


if __name__ == '__main__':
    app.run(debug=True)
