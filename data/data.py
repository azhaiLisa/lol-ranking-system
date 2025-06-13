import requests
import json
import time
import pickle
import atexit

def save_progress():
    with open("matches.pkl", "wb") as f:
        pickle.dump(all_timelines, f)
    with open("seen_ids.pkl", "wb") as f:
        pickle.dump({
            "seen_match_ids": seen_match_ids,
            "seen_puuids": seen_puuids
        }, f)
    print("Progress saved.")

# Register auto-save on exit
atexit.register(save_progress)

API_KEY = 'RGAPI-xxx' # Use your own key
HEADERS = {'X-Riot-Token': API_KEY}

# Match targets
TOTAL_MATCHES = 2000

REGION_CONFIG = {
    'kr':   {'route': 'asia',     'weight': 1.8},
    'euw1': {'route': 'europe',   'weight': 1.6},
    'na1':  {'route': 'americas', 'weight': 1.2},
    'eun1': {'route': 'europe',   'weight': 1.0},
    'br1':  {'route': 'americas', 'weight': 0.9},
    'jp1':  {'route': 'asia',     'weight': 0.6},
    'la1':  {'route': 'americas', 'weight': 0.5},
    'la2':  {'route': 'americas', 'weight': 0.5},
    ' ':  {'route': 'europe',   'weight': 0.5},
    'ru':   {'route': 'europe',   'weight': 0.4},
}

def get_urls(region, route):
    return {
        "FEATURED_GAMES_URL": f'https://{region}.api.riotgames.com/lol/spectator/v5/featured-games',
        "MATCH_LIST_URL": f'https://{route}.api.riotgames.com/lol/match/v5/matches/by-puuid/{{puuid}}/ids',
        "MATCH_DETAILS_URL": f'https://{route}.api.riotgames.com/lol/match/v5/matches/{{matchId}}/timeline',
        "MATCH_METADATA_URL": f'https://{route}.api.riotgames.com/lol/match/v5/matches/{{matchId}}',
        "SUMMONER_INFO_URL": f'https://{region}.api.riotgames.com/lol/summoner/v4/summoners/by-puuid/{{puuid}}',
        "RANKED_INFO_URL": f'https://{region}.api.riotgames.com/lol/league/v4/entries/by-summoner/{{summonerId}}'
    }

def safe_request(url, retries=3, backoff=5):
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 120))
                print(f"Rate limit hit. Sleeping for {retry_after} seconds...")
                time.sleep(retry_after)
            else:
                print(f"[HTTP {response.status_code}] {url}")
                break
        except requests.exceptions.RequestException as e:
            print(f"[Connection Error] {e} — Retrying in {backoff}s...")
            time.sleep(backoff)
    return None

def extract_solo_rank(ranked_info):
    for entry in ranked_info:
        if entry['queueType'] == 'RANKED_SOLO_5x5':
            return f"{entry['tier']} {entry['rank']}"
    return "UNRANKED"

def save_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Saved {len(data)} matches to {filename}")

# ---- MAIN ----
try:
    all_timelines = []
    total_weight = sum(cfg['weight'] for cfg in REGION_CONFIG.values())
    region_targets = {
        region: int((cfg['weight'] / total_weight) * TOTAL_MATCHES)
        for region, cfg in REGION_CONFIG.items()
    }

    seen_puuids = set()
    seen_match_ids = set()

    for region, cfg in REGION_CONFIG.items():
        route = cfg['route']
        urls = get_urls(region, route)
        target = region_targets[region]
        print(f"\n=== Collecting from {region} ({route}) | Target: {target} matches ===")

        match_count, skipped_unranked, skipped_nonranked = 0, 0, 0

        while match_count < target:
            featured = safe_request(urls["FEATURED_GAMES_URL"])
            if not featured:
                break

            for game in featured.get("gameList", []):
                for p in game["participants"]:
                    puuid = p["puuid"]
                    if puuid in seen_puuids:
                        continue
                    seen_puuids.add(puuid)

                    summoner = safe_request(urls["SUMMONER_INFO_URL"].format(puuid=puuid))
                    if not summoner:
                        continue

                    ranked_info = safe_request(urls["RANKED_INFO_URL"].format(summonerId=summoner["id"]))
                    if not ranked_info:
                        continue

                    rank = extract_solo_rank(ranked_info)
                    if rank == "UNRANKED":
                        skipped_unranked += 1
                        continue

                    match_ids = safe_request(urls["MATCH_LIST_URL"].format(puuid=puuid))
                    if not match_ids:
                        continue

                    for match_id in match_ids[:10]:
                        if match_id in seen_match_ids:
                            continue

                        metadata = safe_request(urls["MATCH_METADATA_URL"].format(matchId=match_id))
                        if not metadata or metadata['info'].get('queueId') != 420:
                            skipped_nonranked += 1
                            continue

                        timeline = safe_request(urls["MATCH_DETAILS_URL"].format(matchId=match_id))
                        if not timeline:
                            continue

                        all_timelines.append({
                            "match_id": match_id,
                            "rank": rank,
                            "timeline": timeline,
                            "metadata": metadata,
                            "puuid": puuid,
                            "region": region
                        })
                        seen_match_ids.add(match_id)
                        match_count += 1

                        print(f"[{region}] Saved match #{match_count}")
                        
                        if match_count >= target:
                            break
                    if match_count >= target:
                        break
                if match_count >= target:
                    break

        print(f"[{region}] DONE — {match_count} matches, {skipped_unranked} unranked players, {skipped_nonranked} non-solo matches.")

    save_to_json(all_timelines, "match_ranked_multiregion.json")
except Exception as e:
    print(f"[FATAL] Crash: {e}")
    save_progress()
    raise