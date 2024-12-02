import requests
import json
from datetime import datetime

API_KEY = 'RGAPI-bb4372de-831b-45ac-8005-6e5ce1f5c637'
FEATURED_GAMES_URL = 'https://na1.api.riotgames.com/lol/spectator/v5/featured-games'
MATCH_LIST_URL = 'https://americas.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids'
MATCH_DETAILS_URL = 'https://americas.api.riotgames.com/lol/match/v5/matches/{matchId}/timeline'

headers = {'X-Riot-Token': API_KEY}

def get_featured_games():
    """Fetch a list of featured games."""
    headers = {'X-Riot-Token': API_KEY}
    response = requests.get(FEATURED_GAMES_URL, headers=headers)
    response.raise_for_status()
    return response.json()

def get_match_ids_by_puuid(puuid):
    url = MATCH_LIST_URL.format(puuid=puuid)
    headers = {'X-Riot-Token': API_KEY}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

def format_timestamp(timestamp):
    """Convert Unix timestamp (in ms) to readable format."""
    return datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')

def get_match_timeline(match_id):
    url = MATCH_DETAILS_URL.format(matchId=match_id)
    headers = {'X-Riot-Token': API_KEY}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

def save_to_json(data, filename):
    """Save data to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Data saved to {filename}")
    
    
# Main logic
try:
    featured_games = get_featured_games()
    print("Featured Games:")
    
    all_timelines = []
    for game in featured_games['gameList']:
        for participant in game['participants']:
            puuid = participant['puuid']
            print(f"Fetching matches for player with PUUID: {puuid}")
            try:
                # Fetch match IDs
                match_ids = get_match_ids_by_puuid(puuid)
                print(f"Found {len(match_ids)} matches for PUUID: {puuid}")
                for match_id in match_ids[:3]:  # Fetch timelines for the first 3 matches as an example
                    timeline = get_match_timeline(match_id)
                    all_timelines.append({
                        "match_id": match_id,
                        "timeline": timeline
                    })
            except requests.exceptions.RequestException as e:
                print(f"Error fetching data for PUUID {puuid}: {e}")

    # Save all timeline data to a file
    save_to_json(all_timelines, filename="match_timelines.json")

except requests.exceptions.RequestException as e:
    print(f"HTTP Request failed: {e}")