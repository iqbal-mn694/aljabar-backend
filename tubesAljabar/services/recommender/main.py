from .SongRecommender import SongRecommender
import json


def main(request):
    recommender = SongRecommender()
    song_name = request.GET.get('song')

    eigenvalues, eigenvectors = recommender.get_eigenvalues_eigenvectors()

    print(eigenvalues.tolist())

    recommendations = recommender.get_recommendations(song_name)
    
    return json.dumps(recommendations, indent=4)
    