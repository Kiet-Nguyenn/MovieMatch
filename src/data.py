import ast
import csv
from collections import Counter
from typing import List, Dict, Set, Optional, Tuple


class Movie:
    """Represents a single movie with its attributes."""
    
    def __init__(self, movie_id: str, title: str, year: Optional[int] = None,
                 rating: float = 0.0, genres: Optional[List[str]] = None,
                 runtime: int = 0, metascore: float = 0.0, 
                 director: Optional[str] = None, cast: Optional[List[str]] = None,
                 gross: float = 0.0, description: str = ""):
        """
        Initialize a Movie object.
        
        Args:
            movie_id: Unique identifier for the movie
            title: Movie title
            year: Release year
            rating: IMDb rating (0-10)
            genres: List of genre tags
            runtime: Movie runtime in minutes
            metascore: Metascore rating (0-100)
            director: Director name
            cast: List of actors
            gross: Box office gross in millions
            description: Movie description/plot
        """
        self.id = movie_id
        self.title = title
        self.year = year
        self.rating = rating
        self.genres = genres if genres else []
        self.runtime = runtime
        self.metascore = metascore
        self.director = director
        self.cast = cast if cast else []
        self.gross = gross
        self.description = description
    
    def __repr__(self) -> str:
        return f"Movie({self.id}, {self.title}, {self.rating})"
    
    def __str__(self) -> str:
        genres_str = ", ".join(self.genres) if self.genres else "Unknown"
        return f"{self.title} ({self.year}) - Rating: {self.rating}/10 - Genres: {genres_str}"
    
    def get_genre_set(self) -> Set[str]:
        """Return genres as a set for efficient comparison."""
        return set(self.genres)


class Dataset:
    """Manages loading and indexing of movies from CSV data."""
    
    def __init__(self):
        """Initialize an empty dataset."""
        self.movies: Dict[str, Movie] = {}
        self.movies_by_id: Dict[str, Movie] = {}
        
        # Index structures for efficient lookup
        self.genres_index: Dict[str, Set[str]] = {}  # genre -> movie_ids
        self.directors_index: Dict[str, Set[str]] = {}  # director -> movie_ids
        self.cast_index: Dict[str, Set[str]] = {}  # actor -> movie_ids
    
    def load_from_csv(self, csv_path: str) -> None:
        """
        Load movies from a CSV file.
        
        Expected CSV columns: Index, Title, Year, Runtime (Minutes), Rating,
        Metascore, Votes, Gross Earnings, Description, Director, Star1, Star2, etc.
        
        Args:
            csv_path: Path to the CSV file
        """
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    movie = self._parse_movie_row(row)
                    if movie:
                        self.movies[movie.id] = movie
                        self.movies_by_id[movie.id] = movie
                        self._update_indices(movie)
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    def _parse_movie_row(self, row: Dict[str, str]) -> Optional[Movie]:
        """
        Parse a single row from CSV and create a Movie object.
        Handles missing/malformed data gracefully.
        """
        try:
            movie_id = row.get('Index', '').strip() or row.get('ID', '').strip()
            title = row.get('Title', '').strip() or row.get('Movie Name', 'Unknown').strip()
            
            if not movie_id or not title:
                return None
            
            # Parse year
            year = None
            try:
                year_str = row.get('Year', '')
                if year_str:
                    year = int(year_str)
                    if year <= 1800 or year > 2100:
                        year = None
            except (ValueError, TypeError):
                year = None
            
            # Parse rating
            rating = 0.0
            try:
                rating = float(row.get('Rating', '0'))
                rating = max(0.0, min(10.0, rating))  # Clamp to 0-10
            except (ValueError, TypeError):
                pass
            
            # Parse runtime
            runtime = 0
            try:
                runtime_str = row.get('Runtime (Minutes)', '') or row.get('Runtime', '')
                runtime_str = runtime_str.replace('min', '').strip()
                runtime = int(float(runtime_str))
                runtime = max(0, runtime)
            except (ValueError, TypeError):
                pass
            
            # Parse metascore
            metascore = 0.0
            try:
                metascore = float(row.get('Metascore', '0'))
                metascore = max(0.0, min(100.0, metascore))
            except (ValueError, TypeError):
                pass
            
            # Parse genres
            genres = []
            genre_str = row.get('Genre', '')
            if genre_str:
                genres = [g.strip() for g in genre_str.split(',') if g.strip()]
            
            # Parse director
            director = row.get('Director', '').strip() or row.get('Directors', '').strip() or None
            if director and director.startswith('[') and director.endswith(']'):
                try:
                    director_list = ast.literal_eval(director)
                    director = director_list[0] if isinstance(director_list, list) and director_list else director
                except (ValueError, SyntaxError):
                    director = director.strip('[]').split(',')[0].strip(" '\"")
            
            # Parse cast
            cast = []
            stars_field = row.get('Stars', '')
            if stars_field:
                if stars_field.startswith('[') and stars_field.endswith(']'):
                    try:
                        cast = [actor.strip() for actor in ast.literal_eval(stars_field)]
                    except (ValueError, SyntaxError):
                        cast = [s.strip().strip("'\"") for s in stars_field.strip('[]').split(',') if s.strip()]
                else:
                    cast = [s.strip() for s in stars_field.split(',') if s.strip()]
            else:
                for i in range(1, 5):
                    star = row.get(f'Star{i}', '').strip()
                    if star:
                        cast.append(star)
            
            # Parse gross
            gross = 0.0
            try:
                gross_str = row.get('Gross Earnings', '') or row.get('Gross', '0')
                gross_str = gross_str.replace(',', '').replace('$', '').strip()
                gross = float(gross_str)
            except (ValueError, TypeError):
                pass
            
            # Parse description
            description = row.get('Description', '').strip() or row.get('Plot', '').strip()
            
            return Movie(
                movie_id=movie_id,
                title=title,
                year=year,
                rating=rating,
                genres=genres,
                runtime=runtime,
                metascore=metascore,
                director=director,
                cast=cast,
                gross=gross,
                description=description
            )
        except Exception:
            return None
    
    def _update_indices(self, movie: Movie) -> None:
        """Update all index structures with a new movie."""
        # Index by genres
        for genre in movie.genres:
            if genre not in self.genres_index:
                self.genres_index[genre] = set()
            self.genres_index[genre].add(movie.id)
        
        # Index by director
        if movie.director:
            if movie.director not in self.directors_index:
                self.directors_index[movie.director] = set()
            self.directors_index[movie.director].add(movie.id)
        
        # Index by cast
        for actor in movie.cast:
            if actor not in self.cast_index:
                self.cast_index[actor] = set()
            self.cast_index[actor].add(movie.id)
    
    def get_movie(self, movie_id: str) -> Optional[Movie]:
        """Retrieve a movie by its ID."""
        return self.movies.get(movie_id)
    
    def get_all_movies(self) -> List[Movie]:
        """Get all movies in the dataset."""
        return list(self.movies.values())
    
    def get_movies_by_genre(self, genre: str) -> List[Movie]:
        """Get all movies with a specific genre."""
        movie_ids = self.genres_index.get(genre, set())
        return [self.movies[mid] for mid in movie_ids if mid in self.movies]
    
    def get_movies_by_director(self, director: str) -> List[Movie]:
        """Get all movies by a specific director."""
        movie_ids = self.directors_index.get(director, set())
        return [self.movies[mid] for mid in movie_ids if mid in self.movies]
    
    def get_movies_by_actor(self, actor: str) -> List[Movie]:
        """Get all movies featuring a specific actor."""
        movie_ids = self.cast_index.get(actor, set())
        return [self.movies[mid] for mid in movie_ids if mid in self.movies]
    
    def get_genres(self) -> Set[str]:
        """Get all unique genres in the dataset."""
        return set(self.genres_index.keys())
    
    def get_directors(self) -> Set[str]:
        """Get all unique directors in the dataset."""
        return set(self.directors_index.keys())
    
    def get_actors(self) -> Set[str]:
        """Get all unique actors in the dataset."""
        return set(self.cast_index.keys())
    
    def get_autocomplete_movie_options(self) -> List[Dict[str, str]]:
        """
        Return {id, label} for each loaded movie with unique display labels.
        Duplicate titles are disambiguated with year or id.
        """
        movies = self.get_all_movies()
        title_counts = Counter(m.title for m in movies)
        options: List[Dict[str, str]] = []
        for m in movies:
            if title_counts[m.title] == 1:
                label = m.title
            else:
                if m.year is not None:
                    label = f"{m.title} ({m.year})"
                else:
                    label = f"{m.title} [{m.id}]"
            options.append({"id": m.id, "label": label})
        label_counts = Counter(o["label"] for o in options)
        for o in options:
            if label_counts[o["label"]] > 1:
                o["label"] = f"{o['label']} [{o['id']}]"
        options.sort(key=lambda x: x["label"].lower())
        return options

    def search_by_title(self, query: str, limit: int = 10) -> List[Movie]:
        """
        Search for movies by title (substring matching).
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
        
        Returns:
            List of matching movies
        """
        query_lower = query.lower()
        results = [
            movie for movie in self.movies.values()
            if query_lower in movie.title.lower()
        ]
        # Sort by rating (highest first)
        results.sort(key=lambda m: (-m.rating, m.title))
        return results[:limit]
    
    def get_statistics(self) -> Dict[str, any]:
        """Get statistics about the dataset."""
        if not self.movies:
            return {
                'total_movies': 0,
                'avg_rating': 0.0,
                'min_rating': 0.0,
                'max_rating': 0.0,
                'avg_runtime': 0,
                'total_genres': 0,
                'total_directors': 0,
                'total_actors': 0
            }
        
        all_movies = self.get_all_movies()
        ratings = [m.rating for m in all_movies if m.rating > 0]
        runtimes = [m.runtime for m in all_movies if m.runtime > 0]
        
        return {
            'total_movies': len(self.movies),
            'avg_rating': sum(ratings) / len(ratings) if ratings else 0.0,
            'min_rating': min(ratings) if ratings else 0.0,
            'max_rating': max(ratings) if ratings else 0.0,
            'avg_runtime': sum(runtimes) / len(runtimes) if runtimes else 0,
            'total_genres': len(self.genres_index),
            'total_directors': len(self.directors_index),
            'total_actors': len(self.cast_index)
        }
    
    def __len__(self) -> int:
        """Return the number of movies in the dataset."""
        return len(self.movies)
    
    def __repr__(self) -> str:
        return f"Dataset(movies={len(self.movies)})"
