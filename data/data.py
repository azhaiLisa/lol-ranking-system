import requests
import json
from datetime import datetime
import time

API_KEY = 'RGAPI-c60af6b3-1f39-4cb4-8882-e4b44bcb8a88'
headers = {'X-Riot-Token': API_KEY}

# PLATFORM_REGIONS = ['na1', 'euw1', 'eun1', 'kr', 'jp1', 'oc1', 'br1', 'la1', 'la2', 'tr1', 'ru']
# REGIONAL_ROUTES = {
#     'na1': 'americas',
#     'br1': 'americas',
#     'la1': 'americas',
#     'la2': 'americas',
#     'euw1': 'europe',
#     'eun1': 'europe',
#     'tr1': 'europe',
#     'ru': 'europe',
#     'kr': 'asia',
#     'jp1': 'asia',
#     'oc1': 'sea'  # Optional, depending on data availability
# }

# def get_urls(platform_region):
#     regional_route = REGIONAL_ROUTES[platform_region]
#     return {
#         "FEATURED_GAMES_URL": f'https://{platform_region}.api.riotgames.com/lol/spectator/v5/featured-games',
#         "MATCH_LIST_URL": f'https://{regional_route}.api.riotgames.com/lol/match/v5/matches/by-puuid/{{puuid}}/ids',
#         "MATCH_DETAILS_URL": f'https://{regional_route}.api.riotgames.com/lol/match/v5/matches/{{matchId}}/timeline',
#         "MATCH_METADATA_URL": f'https://{regional_route}.api.riotgames.com/lol/match/v5/matches/{{matchId}}',
#         "SUMMONER_INFO_URL": f'https://{platform_region}.api.riotgames.com/lol/summoner/v4/summoners/by-puuid/{{puuid}}',
#         "RANKED_INFO_URL": f'https://{platform_region}.api.riotgames.com/lol/league/v4/entries/by-summoner/{{summonerId}}'
#     }

# FEATURED_GAMES_URL = 'https://na1.api.riotgames.com/lol/spectator/v5/featured-games'
# MATCH_LIST_URL = 'https://americas.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids'
# MATCH_DETAILS_URL = 'https://americas.api.riotgames.com/lol/match/v5/matches/{matchId}/timeline'
# MATCH_METADATA_URL = 'https://americas.api.riotgames.com/lol/match/v5/matches/{matchId}'
# SUMMONER_INFO_URL = 'https://na1.api.riotgames.com/lol/summoner/v4/summoners/by-puuid/{puuid}'
# RANKED_INFO_URL = 'https://na1.api.riotgames.com/lol/league/v4/entries/by-summoner/{summonerId}'

# FEATURED_GAMES_URL = 'https://euw1.api.riotgames.com/lol/spectator/v5/featured-games'
# MATCH_LIST_URL = 'https://europe.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids'
# MATCH_DETAILS_URL = 'https://europe.api.riotgames.com/lol/match/v5/matches/{matchId}/timeline'
# MATCH_METADATA_URL = 'https://europe.api.riotgames.com/lol/match/v5/matches/{matchId}'
# SUMMONER_INFO_URL = 'https://euw1.api.riotgames.com/lol/summoner/v4/summoners/by-puuid/{puuid}'
# RANKED_INFO_URL = 'https://euw1.api.riotgames.com/lol/league/v4/entries/by-summoner/{summonerId}'

FEATURED_GAMES_URL = 'https://kr.api.riotgames.com/lol/spectator/v5/featured-games'
MATCH_LIST_URL = 'https://asia.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids'
MATCH_DETAILS_URL = 'https://asia.api.riotgames.com/lol/match/v5/matches/{matchId}/timeline'
MATCH_METADATA_URL = 'https://asia.api.riotgames.com/lol/match/v5/matches/{matchId}'
SUMMONER_INFO_URL = 'https://kr.api.riotgames.com/lol/summoner/v4/summoners/by-puuid/{puuid}'
RANKED_INFO_URL = 'https://kr.api.riotgames.com/lol/league/v4/entries/by-summoner/{summonerId}'

def safe_request(url, headers):
    while True:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 120))
            print(f"Rate limit hit. Sleeping for {retry_after} seconds...")
            time.sleep(retry_after)
        else:
            print(f"Request failed with status {response.status_code}: {response.text}")
            response.raise_for_status()


def get_featured_games():
    return safe_request(FEATURED_GAMES_URL, headers)


def get_match_ids_by_puuid(puuid):
    url = MATCH_LIST_URL.format(puuid=puuid)
    return safe_request(url, headers)


def get_match_timeline(match_id):
    url = MATCH_DETAILS_URL.format(matchId=match_id)
    return safe_request(url, headers)


def get_match_metadata(match_id):
    url = MATCH_METADATA_URL.format(matchId=match_id)
    return safe_request(url, headers)


def get_summoner_info(puuid):
    url = SUMMONER_INFO_URL.format(puuid=puuid)
    return safe_request(url, headers)


def get_ranked_info(summoner_id):
    url = RANKED_INFO_URL.format(summonerId=summoner_id)
    return safe_request(url, headers)


def extract_solo_rank(ranked_info):
    for entry in ranked_info:
        if entry['queueType'] == 'RANKED_SOLO_5x5':
            return f"{entry['tier']} {entry['rank']}"
    return "UNRANKED"


def save_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Data saved to {filename}")


# Main logic
try:
    featured_games = get_featured_games()
    print("Featured Games:")

    all_timelines = []
    seen_puuids = set()

    processed_count = 0
    skipped_unranked = 0
    skipped_nonranked_matches = 0

    NUM_ITERATIONS = 100  # Each batch gets around 100 games
    for i in range(NUM_ITERATIONS):
        print(f"\n=== Featured Game Batch {i+1} ===")
        featured_games = get_featured_games()
        
        for game in featured_games['gameList']:
            for participant in game['participants']:
                puuid = participant['puuid']
                if puuid in seen_puuids:
                    continue
                seen_puuids.add(puuid)

                print(f"\nFetching data for player with PUUID: {puuid}")
                try:
                    # Get summoner info and ranked info
                    summoner = get_summoner_info(puuid)
                    ranked_info = get_ranked_info(summoner['id'])
                    rank = extract_solo_rank(ranked_info)

                    if rank == "UNRANKED":
                        print(f"Skipping player {puuid} — Unranked")
                        skipped_unranked += 1
                        continue

                    processed_count += 1
                    print(f"Player Rank: {rank}")

                    # Fetch match IDs
                    match_ids = get_match_ids_by_puuid(puuid)
                    print(f"Found {len(match_ids)} matches for PUUID: {puuid}")

                    for match_id in match_ids[:10]:  # Limit to first 10 matches
                        metadata = get_match_metadata(match_id)
                        queue_id = metadata['info'].get('queueId', -1)

                        if queue_id != 420:
                            print(f"Skipping match {match_id} — Not Ranked Solo/Duo (queueId={queue_id})")
                            skipped_nonranked_matches += 1
                            continue

                        timeline = get_match_timeline(match_id)
                        all_timelines.append({
                            "match_id": match_id,
                            "rank": rank,
                            "timeline": timeline,
                            "puuid": puuid
                        })

                except requests.exceptions.RequestException as e:
                    print(f"Error fetching data for PUUID {puuid}: {e}")

    # Save all timeline data
    save_to_json(all_timelines, filename="match_ranked_korea.json")

    # Print summary
    print("\n--- Summary ---")
    print(f"Total players processed: {processed_count}")
    print(f"Total players skipped (unranked): {skipped_unranked}")
    print(f"Total matches skipped (not ranked solo): {skipped_nonranked_matches}")
    print(f"Total ranked solo matches saved: {len(all_timelines)}")

except requests.exceptions.RequestException as e:
    print(f"HTTP Request failed: {e}")
