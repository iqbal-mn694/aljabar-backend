from .SongRecommender import SongRecommender
import json


def main(request):
    recommender = SongRecommender()

    song_name = request.GET.get('song')

    eigenvalues, eigenvectors = recommender.get_eigenvalues_eigenvectors()

    # Dapatkan rekomendasi untuk sebuah lagu
    # song_name = 'Viva La Vida'   # Ganti dengan judul lagu dari dataset Anda
    recommendations = recommender.get_recommendations(song_name)

    # print(f"\nRekomendasi untuk {song_name}:")

    return json.dumps(recommendations, indent=4)
    # for song, similarity in recommendations:
    #     return (f"{song}: Tingkat kemiripan = {similarity:.4f}")
        # print(f"{song}: Tingkat kemiripan = {similarity:.4f}")
