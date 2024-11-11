import numpy as np
from sklearn.preprocessing import normalize
import pandas as pd
import os 


class SongRecommender:
    def __init__(self):
        base_dir = os.path.dirname(__file__)  # Direktori file saat ini
        csv_path = os.path.join(base_dir, 'songs.csv')  # Gabungkan path relatif
        music_data = pd.read_csv(csv_path, encoding='ISO-8859-1')
        features = [
           'bpm', 'danceability_%', 'energy_%'
        ]
        features_matrix = music_data[features].values
        # print(music_data.head())

        self.normalized_features = normalize(features_matrix)
        self.similarity_matrix = np.dot(self.normalized_features, self.normalized_features.T)
        self.song_names = music_data['track_name'].tolist()  # Simpan nama lagu
        self.bpm = music_data['bpm'].to_list()
        self.danceability = music_data['danceability_%'].tolist()
        self.energy = music_data['energy_%'].tolist()

    def get_eigenvalues_eigenvectors(self):
        """Menghitung nilai eigen dan vektor eigen dari matriks kemiripan"""
        eigenvalues, eigenvectors = np.linalg.eigh(self.similarity_matrix)
        
        # sorting nilai eigen
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        return eigenvalues, eigenvectors

    def get_recommendations(self, song_name, num_recommendations=2):
        """Memberikan rekomendasi lagu berdasarkan kemiripan"""
        try:
            # Dapatkan indeks lagu yang dicari
            song_idx = self.song_names.index(song_name)
        except ValueError:
            return "Lagu tidak ditemukan dalam database"

        similarities = self.similarity_matrix[song_idx]

        similar_indices = similarities.argsort()[::-1][1:num_recommendations + 1]

    
        recommendations = []
        for idx in similar_indices:
            song = {
                "name": self.song_names[idx],
                "bpm": self.bpm[idx],
                "danceability": self.danceability[idx],
                "energy": self.energy[idx],
                "similarity": similarities[idx]
            }
            
            recommendations.append(song)

        return recommendations
