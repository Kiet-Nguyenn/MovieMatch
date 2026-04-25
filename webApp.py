#!/usr/bin/env python3
import sys
import json
from pathlib import Path

from flask import Flask, render_template, request
from src.data import Dataset

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.recommender import ContentBasedRecommender, PopularityRecommender, HybridRecommender, UserBasedRecommender

app = Flask(__name__)

algorithms = [
    ("Similar Content", ContentBasedRecommender()),
    ("Similar Popularity", PopularityRecommender())
]
dataset = Dataset()
csv_path = Path(__file__).parent / "Top_10000_Movies_IMDb.csv"

try:
    dataset.load_from_csv(str(csv_path))
    print(f"Successfully loaded {len(dataset)} movies")
except Exception as e:
    print(f"Error loading dataset: {e}")

movie_autocomplete_options = dataset.get_autocomplete_movie_options() if len(dataset) else []
movie_id_to_label = {opt["id"]: opt["label"] for opt in movie_autocomplete_options}
genre_options = sorted(dataset.genres_index.keys()) if len(dataset) else []

actor_options = sorted({
    actor
    for movie in dataset.get_all_movies()
    for actor in movie.cast
    if actor
}) if len(dataset) else []

director_options = sorted({
    movie.director
    for movie in dataset.get_all_movies()
    if movie.director
}) if len(dataset) else []

liked_movie_options = [opt["label"] for opt in movie_autocomplete_options]


@app.route("/", methods=["GET", "POST"])
def home():
    results_by_algorithm = {}
    seed_movie = None
    error_message = None
    query = ""

    if request.method == "POST":
        draft_query = request.form.get("query", "").strip()
        movie_id = request.form.get("movie_id", "").strip()
        raw_profile = request.form.get("user_profile", "{}")
        seed_movie = dataset.get_movie(movie_id) if movie_id else None

        try:
            user_profile = json.loads(raw_profile)
        except json.JSONDecodeError:
            user_profile = {
                "genres": [],
                "actors": [],
                "directors": [],
                "likedMovies": []
        }

        if not movie_id and not draft_query:
            pass
        elif not seed_movie:
            error_message = "Choose a movie from the suggestions — only titles from the loaded dataset are allowed."
            query = draft_query
        else:
            query = movie_id_to_label.get(movie_id, seed_movie.title)

            for name, recommender in algorithms:
                try:
                    recommendations = recommender.recommend(seed_movie, dataset, 12, user_profile)

                    if not recommendations:
                        results_by_algorithm[name] = []
                        continue

                    results_by_algorithm[name] =  [
                        {
                            "title": movie.title,
                            "score": f"{score:.4f}",
                            "rating": movie.rating,
                            "genres": ", ".join(movie.genres) if movie.genres else "N/A",
                            "runtime": movie.runtime,
                            "metascore": movie.metascore,
                            "gross": movie.gross,
                            "director": movie.director,
                            "cast": ", ".join(movie.cast) if movie.cast else "N/A",
                            "description": movie.description
                        }
                        for movie, score in recommendations
                    ]

                except Exception as e:
                    print("algorithm failed:", name, e)
                    results_by_algorithm[name] = [f"Error: {e}"]

    selected_movie_id = seed_movie.id if seed_movie else ""

    return render_template(
        "index.html",
        query=query,
        selected_movie_id=selected_movie_id,
        seed_movie=seed_movie,
        results_by_algorithm=results_by_algorithm,
        error_message=error_message,
        movie_autocomplete_options=movie_autocomplete_options,
        genre_options=genre_options,
        actor_options=actor_options,
        director_options=director_options,
        liked_movie_options=liked_movie_options,
    )


if __name__ == "__main__":
    app.run()