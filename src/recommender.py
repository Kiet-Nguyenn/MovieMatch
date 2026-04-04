import math
from typing import List, Dict, Set, Optional, Tuple
from src.data import Movie, Dataset


class Recommender:
    """Base class for movie recommenders."""
    
    def recommend(self, seed_movie: Movie, dataset: Dataset, 
                  num_recommendations: int = 10) -> List[Tuple[Movie, float]]:
        """
        Generate recommendations based on a seed movie.
        
        Args:
            seed_movie: The reference movie for generating recommendations
            dataset: The movie dataset
            num_recommendations: Number of recommendations to return
        
        Returns:
            List of (Movie, similarity_score) tuples sorted by score
        """
        raise NotImplementedError


class ContentBasedRecommender(Recommender):
    """
    Content-based recommender using weighted similarity across multiple features.
    Features: genre, rating, runtime, metascore, popularity (gross earnings).
    """
    
    def __init__(self, genre_weight: float = 0.35, rating_weight: float = 0.25,
                 runtime_weight: float = 0.15, metascore_weight: float = 0.15,
                 popularity_weight: float = 0.10):
        """
        Initialize content-based recommender with feature weights.
        
        Args:
            genre_weight: Weight for genre similarity (0-1)
            rating_weight: Weight for rating similarity (0-1)
            runtime_weight: Weight for runtime similarity (0-1)
            metascore_weight: Weight for metascore similarity (0-1)
            popularity_weight: Weight for popularity similarity (0-1)
        """
        self.genre_weight = genre_weight
        self.rating_weight = rating_weight
        self.runtime_weight = runtime_weight
        self.metascore_weight = metascore_weight
        self.popularity_weight = popularity_weight
        
        # Normalize weights
        total = sum([genre_weight, rating_weight, runtime_weight, 
                    metascore_weight, popularity_weight])
        self.genre_weight /= total
        self.rating_weight /= total
        self.runtime_weight /= total
        self.metascore_weight /= total
        self.popularity_weight /= total
    
    def recommend(self, seed_movie: Movie, dataset: Dataset,
                  num_recommendations: int = 10) -> List[Tuple[Movie, float]]:
        """Generate recommendations based on content similarity."""
        scores = {}
        
        for movie in dataset.get_all_movies():
            if movie.id == seed_movie.id:
                continue
            
            similarity = self._calculate_similarity(seed_movie, movie)
            scores[movie.id] = similarity
        
        # Sort by similarity score and return top recommendations
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(dataset.get_movie(mid), score) 
                for mid, score in ranked[:num_recommendations]]
    
    def _calculate_similarity(self, movie1: Movie, movie2: Movie) -> float:
        """Calculate weighted similarity between two movies."""
        genre_sim = self._genre_similarity(movie1, movie2)
        rating_sim = self._rating_similarity(movie1, movie2)
        runtime_sim = self._runtime_similarity(movie1, movie2)
        metascore_sim = self._metascore_similarity(movie1, movie2)
        popularity_sim = self._popularity_similarity(movie1, movie2)
        
        total_similarity = (
            self.genre_weight * genre_sim +
            self.rating_weight * rating_sim +
            self.runtime_weight * runtime_sim +
            self.metascore_weight * metascore_sim +
            self.popularity_weight * popularity_sim
        )
        
        return total_similarity
    
    def _genre_similarity(self, movie1: Movie, movie2: Movie) -> float:
        """
        Calculate genre similarity using Jaccard similarity.
        Returns: 0-1 (1 = identical genres)
        """
        if not movie1.genres and not movie2.genres:
            return 1.0
        
        set1 = set(movie1.genres)
        set2 = set(movie2.genres)
        
        if not set1 and not set2:
            return 1.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _rating_similarity(self, movie1: Movie, movie2: Movie) -> float:
        """
        Calculate rating similarity.
        Returns: 0-1 (1 = identical or very close ratings)
        """
        # Normalize difference (max difference is 10)
        diff = abs(movie1.rating - movie2.rating)
        return 1.0 - (diff / 10.0)
    
    def _runtime_similarity(self, movie1: Movie, movie2: Movie) -> float:
        """
        Calculate runtime similarity.
        Returns: 0-1 (1 = runtime within 15 minutes)
        """
        if movie1.runtime == 0 or movie2.runtime == 0:
            return 0.5  # Neutral score if data missing
        
        diff = abs(movie1.runtime - movie2.runtime)
        # Similarity decreases as difference increases (max penalty at 120+ min difference)
        penalty = min(diff / 120.0, 1.0)
        return 1.0 - penalty
    
    def _metascore_similarity(self, movie1: Movie, movie2: Movie) -> float:
        """
        Calculate metascore similarity.
        Returns: 0-1 (1 = metascores within 10 points)
        """
        if movie1.metascore == 0 or movie2.metascore == 0:
            return 0.5  # Neutral score if data missing
        
        diff = abs(movie1.metascore - movie2.metascore)
        penalty = min(diff / 100.0, 1.0)
        return 1.0 - penalty
    
    def _popularity_similarity(self, movie1: Movie, movie2: Movie) -> float:
        """
        Calculate popularity similarity based on gross earnings.
        Returns: 0-1 (1 = both high or both low gross)
        """
        if movie1.gross == 0 or movie2.gross == 0:
            return 0.5  # Neutral score if data missing
        
        # Normalize to log scale to handle large ranges
        log1 = math.log10(movie1.gross + 1)
        log2 = math.log10(movie2.gross + 1)
        
        max_log = math.log10(1e9 + 1)  # Assume max is ~$1B
        norm1 = log1 / max_log
        norm2 = log2 / max_log
        
        diff = abs(norm1 - norm2)
        return 1.0 - diff


class PopularityRecommender(Recommender):
    """
    Popularity-based recommender that suggests highly-rated movies
    in the same genres as the seed movie.
    """
    
    def __init__(self, rating_threshold: float = 7.0):
        """
        Initialize popularity recommender.
        
        Args:
            rating_threshold: Minimum rating for recommendations
        """
        self.rating_threshold = rating_threshold
    
    def recommend(self, seed_movie: Movie, dataset: Dataset,
                  num_recommendations: int = 10) -> List[Tuple[Movie, float]]:
        """Generate recommendations based on popularity in same genres."""
        # Filter movies in the same genres
        candidates = set()
        for genre in seed_movie.genres:
            candidates.update(dataset.genres_index.get(genre, set()))
        
        # Remove the seed movie itself
        candidates.discard(seed_movie.id)
        
        # Score candidates: prefer higher rating in same genres
        scores = {}
        for movie_id in candidates:
            movie = dataset.get_movie(movie_id)
            if movie and movie.rating >= self.rating_threshold:
                # Boost score for movies with higher ratings
                score = movie.rating * 0.7 + (movie.metascore / 10.0) * 0.3
                scores[movie_id] = score
        
        # Sort by score and return top recommendations
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(dataset.get_movie(mid), score)
                for mid, score in ranked[:num_recommendations]]


class HybridRecommender(Recommender):
    """
    Hybrid recommender combining content-based and popularity-based approaches.
    """
    
    def __init__(self, content_weight: float = 0.6, popularity_weight: float = 0.4):
        """
        Initialize hybrid recommender.
        
        Args:
            content_weight: Weight for content-based recommendations (0-1)
            popularity_weight: Weight for popularity-based recommendations (0-1)
        """
        self.content_weight = content_weight / (content_weight + popularity_weight)
        self.popularity_weight = popularity_weight / (content_weight + popularity_weight)
        
        self.content_recommender = ContentBasedRecommender()
        self.popularity_recommender = PopularityRecommender()
    
    def recommend(self, seed_movie: Movie, dataset: Dataset,
                  num_recommendations: int = 10) -> List[Tuple[Movie, float]]:
        """Generate hybrid recommendations."""
        # Get content-based recommendations
        content_recs = self.content_recommender.recommend(
            seed_movie, dataset, num_recommendations * 2)
        content_scores = {movie.id: score for movie, score in content_recs}
        
        # Get popularity-based recommendations
        popularity_recs = self.popularity_recommender.recommend(
            seed_movie, dataset, num_recommendations * 2)
        popularity_scores = {movie.id: score for movie, score in popularity_recs}
        
        # Combine scores
        all_movie_ids = set(content_scores.keys()) | set(popularity_scores.keys())
        hybrid_scores = {}
        
        for movie_id in all_movie_ids:
            content_score = content_scores.get(movie_id, 0.0)
            popularity_score = popularity_scores.get(movie_id, 0.0)
            
            # Normalize and combine
            hybrid_scores[movie_id] = (
                self.content_weight * content_score +
                self.popularity_weight * popularity_score
            )
        
        # Sort and return
        ranked = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        return [(dataset.get_movie(mid), score)
                for mid, score in ranked[:num_recommendations]]


class UserBasedRecommender(Recommender):
    """
    Simple user-based collaborative filtering recommender.
    Finds movies liked by users with similar taste (based on genre preferences).
    """
    
    def recommend(self, seed_movie: Movie, dataset: Dataset,
                  num_recommendations: int = 10) -> List[Tuple[Movie, float]]:
        """
        Generate recommendations based on movies that share genres with seed.
        Uses a simple weighted approach based on genre overlap.
        """
        seed_genres = set(seed_movie.genres)
        if not seed_genres:
            return []
        
        scores = {}
        for movie in dataset.get_all_movies():
            if movie.id == seed_movie.id:
                continue
            
            movie_genres = set(movie.genres)
            if not movie_genres:
                continue
            
            # Count matching genres
            matching_genres = len(seed_genres & movie_genres)
            if matching_genres == 0:
                continue
            
            # Jaccard similarity with bias toward rating
            jaccard = matching_genres / len(seed_genres | movie_genres)
            rating_factor = movie.rating / 10.0  # Normalize rating
            
            score = jaccard * 0.7 + rating_factor * 0.3
            scores[movie.id] = score
        
        # Sort and return
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(dataset.get_movie(mid), score)
                for mid, score in ranked[:num_recommendations]]
