import trueskill
import pandas as pd


def update_hero_mmr(hero, time, mu, sigma, hero_mmr):
    #untested
    if hero in hero_mmr:
        hero_mmr[hero]["history"]["update_time"].append(hero_mmr[hero]["update_time"])
        hero_mmr[hero]["history"]["mu"].append(hero_mmr[hero]["mu"])
        hero_mmr[hero]["history"]["sigma"].append(hero_mmr[hero]["sigma"])
    else:
        hero_mmr[hero] = {"history": {"update_time": [], "mu": [], "sigma": []}}
    hero_mmr[hero].update({"update_time": time, "mu": mu, "sigma": sigma})
    return hero_mmr

def rate_game(game_time, heroes, winner, hero_mmr):
    ratings = []
    for hero in heroes:
        if hero in hero_mmr:
            mu = hero_mmr[hero]["mu"]
            sigma = hero_mmr[hero]["sigma"]
            r = trueskill.Rating(mu = mu, sigma = sigma)
        else:
            r = trueskill.Rating()
            
        ratings.append(r)
    team1 = dict(zip(heroes[:5], ratings[:5]))
    team2 = dict(zip(heroes[5:], ratings[5:]))
    
    rankings = [0,0]
    rankings[winner] = 1
        
    rated_groups = trueskill.rate([team1,team2], rankings)
    
    for hero in team1:
        new_r = rated_groups[0][hero]
        hero_mmr = update_hero_mmr(hero, game_time, new_r.mu, new_r.sigma, hero_mmr)
        
    for hero in team2:
        new_r = rated_groups[1][hero]
        hero_mmr = update_hero_mmr(hero, game_time, new_r.mu, new_r.sigma, hero_mmr)
    
    return hero_mmr


replays = pd.read_csv("../data/processed/teams_20181001-20190123_processed.csv", nrows = 1000)
hero_mmr = {}
for index, replay in replays.iterrows():
    game_time = replay["game_date"]
    heroes = replay[["hero"+str(i) for i in range(1,11)]]
    winner = replay["winner"]
    hero_mmr = rate_game(game_time, heroes, winner, hero_mmr)
