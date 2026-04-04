#!/usr/bin/env python3
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data import Dataset
from src.recommender import ContentBasedRecommender, PopularityRecommender, HybridRecommender, UserBasedRecommender


def main():
    """Demonstrate Checkpoint 2: Recommendation Algorithms"""

    print("=" * 60)
    print("Movie Recommendation Algorithms")
    print("=" * 60)

    # Load dataset
    print("\nLoading movie dataset...")
    dataset = Dataset()
    csv_path = Path(__file__).parent / "Top_10000_Movies_IMDb.csv"

    try:
        dataset.load_from_csv(str(csv_path))
        print(f"Successfully loaded {len(dataset)} movies")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Get a seed movie for testing
    seed_movies = dataset.search_by_title("Inception")
    if not seed_movies:
        # Fallback to highest rated movie
        all_movies = dataset.get_all_movies()
        seed_movie = sorted(all_movies, key=lambda m: m.rating, reverse=True)[0]
    else:
        seed_movie = seed_movies[0]

    print(f"\nUsing seed movie: {seed_movie.title}")
    print(f"Rating: {seed_movie.rating}/10")
    print(f"Genres: {', '.join(seed_movie.genres)}")

    # Test each recommendation algorithm
    algorithms = [
        ("Content-Based", ContentBasedRecommender()),
        ("Popularity-Based", PopularityRecommender()),
        ("Hybrid (60% Content + 40% Popularity)", HybridRecommender()),
        ("User-Based (Collaborative)", UserBasedRecommender())
    ]

    print("\n" + "=" * 60)
    print("RECOMMENDATION ALGORITHMS TEST")
    print("=" * 60)

    for name, recommender in algorithms:
        print(f"\n{name} Recommendations:")
        print("-" * 40)

        try:
            recommendations = recommender.recommend(seed_movie, dataset, 5)

            if not recommendations:
                print("  No recommendations generated")
                continue

            for i, (movie, score) in enumerate(recommendations, 1):
                print(f"  {i:2d}. {movie.title:45s} Score: {score:.4f}")
        except Exception as e:
            print(f"  Error: {e}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()