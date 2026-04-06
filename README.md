# MovieMatch

## Setup Instructions

### Requirements
- Python 3.8 or higher
- Flask 3.1.3 or higher

### Running the Demo
```bash
cd MovieMatch
python webApp.py
```
Put in any browser http://127.0.0.1:5000

In this demo the web app will:
1. Load the movie dataset (Top_10000_Movies_IMDb.csv)
2. Start a local server for website 
3. Demonstrate all 4 recommendation algorithms working
4. Show sample recommendations for each algorithm

## Project Structure
- `src/data.py` - Movie data loading and management
- `src/recommender.py` - Four recommendation algorithms
- `static/index.css` - Style file for index.html
- `templates/index.html` - MovieMatch homepage html
- `webApp.py` - Website backend that runs recommendation algorithms and sends results to HTML template for display 
- `checkpoint2_demo.py` - old script file to be removed 
- `Top_10000_Movies_IMDb.csv` - Movie dataset

## Algorithms Demonstrated
1. Content-Based Recommender
2. Popularity-Based Recommender
3. Hybrid Recommender (60% content + 40% popularity)
4. User-Based Recommender (collaborative filtering)
