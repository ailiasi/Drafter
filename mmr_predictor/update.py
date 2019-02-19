import trueskill
import pandas as pd
import numpy as np

env = trueskill.setup(draw_probability = 0)

def update_hero_mmr(hero, time, mu, sigma, ranking, hero_mmr):
    if hero in hero_mmr:
        hero_mmr[hero]["history"]["update_time"].append(time)
        hero_mmr[hero]["history"]["mu"].append(mu)
        hero_mmr[hero]["history"]["sigma"].append(sigma)
        hero_mmr[hero]["history"]["ranking"].append(ranking)
    else:
        hero_mmr[hero] = {"history": 
                            {"update_time": [time], 
                             "mu": [mu], 
                             "sigma": [sigma], 
                             "ranking":[ranking]}}
    hero_mmr[hero].update({"update_time": time, "mu": mu, "sigma": sigma})
    return hero_mmr

def win_probability(ratings1, ratings2):
    delta_mu = sum(r.mu for r in ratings1) - sum(r.mu for r in ratings2)
    sum_sigma = sum(r.sigma ** 2 for r in ratings1 + ratings2)
    size = len(ratings1) + len(ratings2)
    denom = np.sqrt(size * (trueskill.BETA * trueskill.BETA) + sum_sigma)
    return env.cdf(delta_mu / denom)

def win_probability_heroes(team1, team2, mmr_dict):
    ratings1 = []
    for hero in team1:
        r = env.Rating(mmr_dict[hero]["mu"], mmr_dict[hero]["sigma"])
        ratings1.append(r)
    ratings2 = []
    for hero in team2:
        r = env.Rating(mmr_dict[hero]["mu"], mmr_dict[hero]["sigma"])
        ratings2.append(r)
    return win_probability(ratings1, ratings2)

def rate_game(game_time, heroes, winner, hero_mmr):
    ratings = []
    for hero in heroes:
        if hero in hero_mmr:
            mu = hero_mmr[hero]["mu"]
            sigma = hero_mmr[hero]["sigma"]
            r = env.Rating(mu = mu, sigma = sigma)
        else:
            r = env.Rating()
            
        ratings.append(r)
    team1 = dict(zip(heroes[:5], ratings[:5]))
    team2 = dict(zip(heroes[5:], ratings[5:]))
    
    rankings = [1,1]
    rankings[winner] = 0
        
    rated_groups = env.rate([team1,team2], rankings)
    
    for hero in team1:
        new_r = rated_groups[0][hero]
        hero_mmr = update_hero_mmr(hero, game_time, new_r.mu, new_r.sigma, rankings[0], hero_mmr)
        
    for hero in team2:
        new_r = rated_groups[1][hero]
        hero_mmr = update_hero_mmr(hero, game_time, new_r.mu, new_r.sigma, rankings[0], hero_mmr)
    
    return hero_mmr

def read_replays(filename, game_type, game_version):
    print("Reading replays...")
    replays = pd.read_csv(filename, parse_dates=["game_date"])
    data = replays[(replays["game_type"] == game_type) & (replays["game_version"].str.startswith(game_version))]
    data = data.sort_values("game_date")
    return data

def get_win_rate(replays, hero):
    herocols =  ["hero"+str(i) for i in range(1,11)]
    team0 = ["hero"+str(i) for i in range(1,6)]
    team1 = ["hero"+str(i) for i in range(6,11)]
    
    games = replays[(replays[herocols] == hero).any(axis = 1)]
    n_games = len(games)
    
    team0_wins = games[((games[team0] == hero).any(axis = 1)) & (games["winner"] == 0)]
    n_team0_wins = len(team0_wins)
    
    team1_wins = games[((games[team1] == hero).any(axis = 1)) & (games["winner"] == 1)]
    n_team1_wins = len(team1_wins)
    
    return((n_team0_wins + n_team1_wins)/n_games)
    

def calculate_mmr(replays):
    mmr_dict = {}
    print("Calculating mmr...")
    for index, replay in replays.iterrows():
        game_time = replay["game_date"]
        heroes = replay[["hero"+str(i) for i in range(1,11)]]
        winner = replay["winner"]
        mmr_dict = rate_game(game_time, heroes, winner, mmr_dict)
    return mmr_dict

def list_mmr(mmr_dict):
    mmr_list = []
    for hero in mmr_dict:
        n_wins = mmr_dict[hero]["history"]["ranking"].count(0)
        n_games = len(mmr_dict[hero]["history"]["ranking"])
        wr = n_wins/n_games
        mmr_list.append([hero, mmr_dict[hero]["mu"], mmr_dict[hero]["sigma"], wr, n_games])
    mmr_list = pd.DataFrame(mmr_list, columns = ["hero", "mu", "sigma", "wr", "n"])
    return mmr_list

replays = read_replays("../data/processed/teams_20181001-20190123_processed.csv", "HeroLeague", "2.4")
cut = int(len(replays)*0.8)
replays_train = replays.iloc[:cut]
replays_test = replays.iloc[cut:]

mmr_dict = calculate_mmr(replays_train)
mmr_list = list_mmr(mmr_dict)
    
res = []
for i, row in replays_test.iterrows():
    team1 = row[["hero"+str(i) for i in range(1,6)]]
    team2 = row[["hero"+str(i) for i in range(6,11)]]
    wp = win_probability_heroes(team1, team2, mmr_dict)
    pred_win = 0 if wp > 0.5 else 1
    res.append((pred_win,row["winner"]))
    
correct = 0
total = 0
for pred_w, w in res:
    if pred_w == w:
        correct += 1
    total += 1
    
print("accuracy: " + str(correct/total))