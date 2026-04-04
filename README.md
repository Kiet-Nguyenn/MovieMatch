# MovieMatch

## Setup Instructions

### Requirements
- Python 3.8 or higher
- No external dependencies required (uses only Python standard library)

### Running the Demo
```bash
cd MovieMatch
python checkpoint2_demo.py
```

This will:
1. Load the movie dataset (Top_10000_Movies_IMDb.csv)
2. Demonstrate all 4 recommendation algorithms working
3. Show sample recommendations for each algorithm

## Project Structure
- `src/data.py` - Movie data loading and management
- `src/recommender.py` - Four recommendation algorithms
- `checkpoint2_demo.py` - Simple demonstration script
- `Top_10000_Movies_IMDb.csv` - Movie dataset

## Algorithms Demonstrated
1. Content-Based Recommender
2. Popularity-Based Recommender
3. Hybrid Recommender (60% content + 40% popularity)
4. User-Based Recommender (collaborative filtering)
