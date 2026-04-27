import math
from typing import List, Dict, Set, Optional, Tuple
from src.data import Movie, Dataset

def normalize_text(value: str) -> str:
    return (value or "").strip().lower()


def apply_user_preferences(movie: Movie, user_profile: Optional[Dict], base_score: float) -> float:
    if not user_profile:
        return base_score

    bonus = 0.0

    preferred_genres = {normalize_text(g) for g in user_profile.get("genres", [])}
    preferred_actors = {normalize_text(a) for a in user_profile.get("actors", [])}
    preferred_directors = {normalize_text(d) for d in user_profile.get("directors", [])}
    liked_movies = {normalize_text(m) for m in user_profile.get("likedMovies", [])}

    movie_genres = {normalize_text(g) for g in movie.genres}
    movie_cast = {normalize_text(a) for a in movie.cast}
    movie_director = normalize_text(movie.director)
    movie_title = normalize_text(movie.title)

    if preferred_genres and movie_genres.intersection(preferred_genres):
        bonus += 0.15

    if preferred_actors and movie_cast.intersection(preferred_actors):
        bonus += 0.10

    if preferred_directors and movie_director in preferred_directors:
        bonus += 0.10

    if liked_movies and movie_title in liked_movies:
        bonus += 0.20

    return base_score + bonus

def normalize_scores(scores):
        if not scores:
            return scores

        min_score = min(scores.values())
        max_score = max(scores.values())

        if max_score == min_score:
            return {k: 0.0 for k in scores}

        return {
            k: (v - min_score) / (max_score - min_score)
            for k, v in scores.items()
        }

class Recommender:
    """Base class for movie recommenders."""
    
    def recommend(self, seed_movie: Movie, dataset: Dataset, 
                  num_recommendations: int = 10, user_profile: Optional[Dict] = None) -> List[Tuple[Movie, float]]:
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
    
    def __init__(self, genre_weight: float = 0.50, rating_weight: float = 0.2,
                 runtime_weight: float = 0.1, metascore_weight: float = 0.05,
                 popularity_weight: float = 0.15):
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
                  num_recommendations: int = 10, user_profile: Optional[Dict] = None) -> List[Tuple[Movie, float]]:
        """Generate recommendations based on content similarity."""
        scores = {}
        
        for movie in dataset.get_all_movies():
            if movie.id == seed_movie.id:
                continue
            
            similarity = self._calculate_similarity(seed_movie, movie)
            similarity = apply_user_preferences(movie, user_profile, similarity)

            if similarity < 0.3:
                continue

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
            return 0.0  # Neutral score if data missing
        
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
                  num_recommendations: int = 10, user_profile: Optional[Dict] = None) -> List[Tuple[Movie, float]]:
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
                score = apply_user_preferences(movie, user_profile, score)
                scores[movie_id] = score
        
        # Sort by score and return top recommendations
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(dataset.get_movie(mid), score)
                for mid, score in ranked[:num_recommendations]]


class HybridRecommender(Recommender):
    """
    Hybrid recommender combining content-based and popularity-based approaches.
    """
    
    def __init__(self, content_weight: float = 0.4,  bm25_weight: float = 0.4, popularity_weight: float = 0.2):
        """
        Initialize hybrid recommender.
        
        Args:
            content_weight: Weight for content-based recommendations (0-1)
            bm25_weight: Weight for BM25 based recommendations (0-1)
            popularity_weight: Weight for popularity-based recommendations (0-1)
        """
        total = content_weight + bm25_weight + popularity_weight

        self.content_weight = content_weight / total
        self.bm25_weight = bm25_weight / total
        self.popularity_weight = popularity_weight / total

        self.content_recommender = ContentBasedRecommender()
        self.bm25_recommender = BM25DescriptionRecommender()
        self.popularity_recommender = PopularityRecommender()
    
    def recommend(self, seed_movie: Movie, dataset: Dataset,
              num_recommendations: int = 10,
              user_profile: Optional[Dict] = None) -> List[Tuple[Movie, float]]:

        content_recs = self.content_recommender.recommend(
            seed_movie, dataset, num_recommendations * 2, user_profile
        )
        bm25_recs = self.bm25_recommender.recommend(
            seed_movie, dataset, num_recommendations * 2, user_profile
        )
        popularity_recs = self.popularity_recommender.recommend(
            seed_movie, dataset, num_recommendations * 2, user_profile
        )

        content_scores = {movie.id: score for movie, score in content_recs}
        bm25_scores = {movie.id: score for movie, score in bm25_recs}
        popularity_scores = {movie.id: score for movie, score in popularity_recs}

        content_scores = normalize_scores(content_scores)
        bm25_scores = normalize_scores(bm25_scores)
        popularity_scores = normalize_scores(popularity_scores)
        

        all_movie_ids = (
            set(content_scores.keys())
            | set(bm25_scores.keys())
            | set(popularity_scores.keys())
        )

        hybrid_scores = {}

        for movie_id in all_movie_ids:
            movie = dataset.get_movie(movie_id)

            if not movie:
                continue

            if not (set(seed_movie.genres) & set(movie.genres)):
                continue

            hybrid_scores[movie_id] = (
                self.content_weight * content_scores.get(movie_id, 0.0)
                + self.bm25_weight * bm25_scores.get(movie_id, 0.0)
                + self.popularity_weight * popularity_scores.get(movie_id, 0.0)
            )

        ranked = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)

        return [
            (dataset.get_movie(movie_id), score)
            for movie_id, score in ranked[:num_recommendations]
        ]

class TFIDFDescriptionRecommender(Recommender):
    """
    Content-based recommender using TF-IDF on movie descriptions
    and cosine similarity.
    """

    def recommend(self, seed_movie: Movie, dataset: Dataset,
                  num_recommendations: int = 10,
                  user_profile: Optional[Dict] = None) -> List[Tuple[Movie, float]]:

        movies = dataset.get_all_movies()
        documents = {
            movie.id: normalize_text(movie.description)
            for movie in movies
            if movie.description
        }

        if seed_movie.id not in documents:
            return []

        tfidf_vectors = self._build_tfidf_vectors(documents)
        seed_vector = tfidf_vectors[seed_movie.id]

        scores = {}

        for movie in movies:
            if movie.id == seed_movie.id:
                continue

            movie_vector = tfidf_vectors.get(movie.id)
            if not movie_vector:
                continue

            similarity = self._cosine_similarity(seed_vector, movie_vector)
            similarity = apply_user_preferences(movie, user_profile, similarity)

            if similarity < 0.1:  # lower threshold for text
                continue

            scores[movie.id] = similarity

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return [
            (dataset.get_movie(movie_id), score)
            for movie_id, score in ranked[:num_recommendations]
        ]

    def _tokenize(self, text: str) -> List[str]:
        return [
            word.strip(".,!?;:()[]{}\"'")
            for word in text.lower().split()
            if word.strip(".,!?;:()[]{}\"'")
        ]

    def _build_tfidf_vectors(self, documents: Dict[int, str]) -> Dict[int, Dict[str, float]]:
        tokenized_docs = {
            movie_id: self._tokenize(text)
            for movie_id, text in documents.items()
        }

        num_docs = len(tokenized_docs)

        # Document frequency
        df = {}
        for tokens in tokenized_docs.values():
            unique_terms = set(tokens)
            for term in unique_terms:
                df[term] = df.get(term, 0) + 1

        vectors = {}

        for movie_id, tokens in tokenized_docs.items():
            tfidf = {}
            total_terms = len(tokens)

            if total_terms == 0:
                vectors[movie_id] = {}
                continue

            for term in tokens:
                tf = tokens.count(term) / total_terms
                idf = math.log((num_docs + 1) / (df[term] + 1)) + 1
                tfidf[term] = tf * idf

            vectors[movie_id] = tfidf

        return vectors

    def _cosine_similarity(self, vector1: Dict[str, float], vector2: Dict[str, float]) -> float:
        common_terms = set(vector1.keys()) & set(vector2.keys())

        dot_product = sum(vector1[term] * vector2[term] for term in common_terms)

        magnitude1 = math.sqrt(sum(weight ** 2 for weight in vector1.values()))
        magnitude2 = math.sqrt(sum(weight ** 2 for weight in vector2.values()))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

class BM25DescriptionRecommender(Recommender):
    """
    Content-based recommender using BM25 on movie descriptions.
    The selected movie description acts as the query.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def recommend(self, seed_movie: Movie, dataset: Dataset,
                  num_recommendations: int = 10,
                  user_profile: Optional[Dict] = None) -> List[Tuple[Movie, float]]:

        movies = dataset.get_all_movies()

        documents = {
            movie.id: normalize_text(movie.description)
            for movie in movies
            if movie.description
        }

        if seed_movie.id not in documents:
            return []

        tokenized_docs = {
            movie_id: self._tokenize(text)
            for movie_id, text in documents.items()
        }

        query_terms = tokenized_docs[seed_movie.id]

        if not query_terms:
            return []

        idf = self._calculate_idf(tokenized_docs)
        avg_doc_length = sum(len(tokens) for tokens in tokenized_docs.values()) / len(tokenized_docs)

        scores = {}

        for movie in movies:
            if movie.id == seed_movie.id:
                continue

            doc_terms = tokenized_docs.get(movie.id)

            if not doc_terms:
                continue

            score = self._bm25_score(query_terms, doc_terms, idf, avg_doc_length)
            score = apply_user_preferences(movie, user_profile, score)

            scores[movie.id] = score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return [
            (dataset.get_movie(movie_id), score)
            for movie_id, score in ranked[:num_recommendations]
        ]

    def _tokenize(self, text: str) -> List[str]:
        return [
            word.strip(".,!?;:()[]{}\"'")
            for word in text.lower().split()
            if word.strip(".,!?;:()[]{}\"'")
        ]

    def _calculate_idf(self, tokenized_docs: Dict[int, List[str]]) -> Dict[str, float]:
        num_docs = len(tokenized_docs)
        df = {}

        for tokens in tokenized_docs.values():
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1

        idf = {}

        for term, freq in df.items():
            idf[term] = math.log((num_docs - freq + 0.5) / (freq + 0.5) + 1)

        return idf

    def _bm25_score(self, query_terms: List[str], doc_terms: List[str],
                    idf: Dict[str, float], avg_doc_length: float) -> float:

        score = 0.0
        doc_length = len(doc_terms)
        term_frequencies = {}

        for term in doc_terms:
            term_frequencies[term] = term_frequencies.get(term, 0) + 1

        for term in set(query_terms):
            if term not in term_frequencies:
                continue

            tf = term_frequencies[term]

            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * (doc_length / avg_doc_length)
            )

            score += idf.get(term, 0.0) * (numerator / denominator)

        return score
