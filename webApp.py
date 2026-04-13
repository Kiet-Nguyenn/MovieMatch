#!/usr/bin/env python3
import sys
from pathlib import Path

from flask import Flask, render_template, request
from src.data import Dataset

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.recommender import ContentBasedRecommender, PopularityRecommender, HybridRecommender, UserBasedRecommender

app = Flask(__name__)

algorithms = [
    ("Similar Content", ContentBasedRecommender()),
    ("Similar Popularity", PopularityRecommender()),
    ("Hybrid (60% Content + 40% Popularity)", HybridRecommender()),
    ("User-Based (Collaborative)", UserBasedRecommender())
]
dataset = Dataset()
csv_path = Path(__file__).parent / "Top_10000_Movies_IMDb.csv"

try:
    dataset.load_from_csv(str(csv_path))
    print(f"Successfully loaded {len(dataset)} movies")
except Exception as e:
    print(f"Error loading dataset: {e}")

@app.route("/", methods=["GET", "POST"])
def home():
    results_by_algorithm = {}
    seed_movie = None
    error_message = None
    query = ""

    if request.method == "POST":
        query = request.form.get("query", "").strip()

        if query:
            seed_movies = dataset.search_by_title(query)

            if not seed_movies:
                all_movies = dataset.get_all_movies()
                seed_movie = sorted(all_movies, key=lambda m: m.rating, reverse=True)[0]
            else:
                seed_movie = seed_movies[0]

            for name, recommender in algorithms:
                try:
                    recommendations = recommender.recommend(seed_movie, dataset, 5)

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

    return render_template(
        "index.html",
        query=query,
        seed_movie=seed_movie,
        results_by_algorithm=results_by_algorithm,
        error_message=error_message
    )


if __name__ == "__main__":
    app.run(debug=True)