import numpy as np
from sklearn.preprocessing import normalize
import pandas as pd
import os 


class SongRecommender:
    def __init__(self):
        base_dir = os.path.dirname(__file__)  
        csv_path = os.path.join(base_dir, 'songs2.csv')
        music_data = pd.read_csv(csv_path, encoding='ISO-8859-1', nrows=3000)

        # menetukan fitur yang akan dicari kemiripannya
        features = [
            'danceability', 'energy', 'loudness'
        ]

        features_matrix = music_data[features].values


        self.normalized_features = normalize(features_matrix) # menormalkan data jika ada data anomali
        
        self.similarity_matrix = np.dot(self.normalized_features, self.normalized_features.T) # menghitung nilai cosine similarity dari lagu-lagu di dataset
        
        # mengambil data data lagu yang diperlukan dari dataset 
        self.song_names = music_data['track_name'].tolist() 
        self.artist_names = music_data['track_artist'].tolist()
        self.release_date = music_data['track_album_release_date'].tolist()
        self.liveness = music_data['liveness'].to_list()
        self.danceability = music_data['danceability'].tolist()
        self.energy = music_data['energy'].tolist()

    # Menghitung nilai eigen dan vektor eigen dari hasil perhitungan cosine
    def get_eigenvalues_eigenvectors(self):
        eigenvalues, eigenvectors = np.linalg.eigh(self.similarity_matrix)
        
        # menyortir nilai eigen
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        return eigenvalues, eigenvectors

    # mengirimkan data ke client side
    def get_recommendations(self, song_name, num_recommendations=5):
        # Memberikan rekomendasi lagu berdasarkan kemiripan dari fitur fitur yang diambil
        try:
            # mendapatkan index lagu yang dicari rekomendasinya
            song_idx = [name.lower() for name in self.song_names].index(song_name.lower())
        except ValueError:
            return "Lagu tidak ditemukan di dalam dataset"

        similarities = self.similarity_matrix[song_idx]

        similar_indices = similarities.argsort()[::-1][1:num_recommendations + 1]

    
        recommendations = []
        for idx in similar_indices:
            song = {
                "name": self.song_names[idx],
                "artist": self.artist_names[idx],
                "release": self.release_date[idx],
                "liveness": self.liveness[idx],
                "danceability": self.danceability[idx],
                "energy": self.energy[idx],
                "similarity": similarities[idx]
            }
            
            recommendations.append(song)

        return recommendations
