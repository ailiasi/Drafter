import trueskill
import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid

HEROCOLS =  ["hero"+str(i) for i in range(1,11)]
TEAM0 = ["hero"+str(i) for i in range(1,6)]
TEAM1 = ["hero"+str(i) for i in range(6,11)]


def update_mmr_dict(hero, time, patch, mu, sigma, ranking, mmr_dict):            
    if hero in mmr_dict:
        mmr_dict[hero]["history"]["update_time"].append(time)
        mmr_dict[hero]["history"]["mu"].append(mu)
        mmr_dict[hero]["history"]["sigma"].append(sigma)
        mmr_dict[hero]["history"]["ranking"].append(ranking)
        
        if patch not in mmr_dict[hero]["history"]["patches"]:
            mmr_dict[hero]["history"]["patches"].append(patch)
    else:
        mmr_dict[hero] = {"history": 
                            {"update_time": [time], 
                             "mu": [mu], 
                             "sigma": [sigma], 
                             "ranking":[ranking],
                             "patches":[patch]}}
    mmr_dict[hero].update({"update_time": time, "mu": mu, "sigma": sigma})
    return mmr_dict

def win_probability(ratings1, ratings2, env = trueskill.setup()):
    delta_mu = sum(r.mu for r in ratings1) - sum(r.mu for r in ratings2)
    sum_sigma = sum(r.sigma ** 2 for r in ratings1 + ratings2)
    size = len(ratings1) + len(ratings2)
    denom = np.sqrt(size * (env.beta * env.beta) + sum_sigma)
    return env.cdf(delta_mu / denom)

def win_probability_heroes(team1, team2, mmr_dict, env = trueskill.setup()):
    ratings1 = []
    for hero in team1:
        r = env.Rating(mmr_dict[hero]["mu"], mmr_dict[hero]["sigma"])
        ratings1.append(r)
    ratings2 = []
    for hero in team2:
        r = env.Rating(mmr_dict[hero]["mu"], mmr_dict[hero]["sigma"])
        ratings2.append(r)
    return win_probability(ratings1, ratings2, env)

def rate_game(game_time, patch, heroes, winner, mmr_dict, env = trueskill.setup()):
    ratings = []
    for hero in heroes:
        if hero in mmr_dict:
            mu = mmr_dict[hero]["mu"]
            if patch not in mmr_dict[hero]["history"]["patches"]:
                sigma = env.sigma
            else:
                sigma = mmr_dict[hero]["sigma"]
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
        mmr_dict = update_mmr_dict(hero, game_time, patch, new_r.mu, new_r.sigma, rankings[0], mmr_dict)
        
    for hero in team2:
        new_r = rated_groups[1][hero]
        mmr_dict = update_mmr_dict(hero, game_time, patch, new_r.mu, new_r.sigma, rankings[0], mmr_dict)
    
    return mmr_dict

def read_replays(filename, game_type, game_version):
    print("Reading replays...")
    replays = pd.read_csv(filename, parse_dates=["game_date"])
    data = replays[(replays["game_type"] == game_type) & (replays["game_version"].str.startswith(game_version))]
    data = data.sort_values("game_date")
    return data

def get_win_rate(replays, hero):    
    games = replays[(replays[HEROCOLS] == hero).any(axis = 1)]
    n_games = len(games)
    
    team0_wins = games[((games[TEAM0] == hero).any(axis = 1)) & (games["winner"] == 0)]
    n_team0_wins = len(team0_wins)
    
    team1_wins = games[((games[TEAM1] == hero).any(axis = 1)) & (games["winner"] == 1)]
    n_team1_wins = len(team1_wins)
    
    return((n_team0_wins + n_team1_wins)/n_games)
    

def calculate_mmr(replays, env = trueskill.setup()):
    mmr_dict = {}
    print("Calculating mmr...")
    for index, replay in replays.iterrows():
        game_time = replay["game_date"]
        patch = replay["game_version"]
        heroes = replay[["hero"+str(i) for i in range(1,11)]]
        winner = replay["winner"]
        mmr_dict = rate_game(game_time, patch, heroes, winner, mmr_dict, env)
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

def accuracy(replays, mmr_dict, env = trueskill.setup()):
    correct = 0
    total = 0
    
    for i, row in replays.iterrows():
        team1 = row[TEAM0]
        team2 = row[TEAM1]
        wp = win_probability_heroes(team1, team2, mmr_dict, env)
        pred_win = 0 if wp > 0.5 else 1
        
        if pred_win == row["winner"]:
            correct += 1
        total += 1

    return correct/total

def binary_crossentropy(replays, mmr_dict, env):
    crossentropy = 0
    
    for i, row in replays.iterrows():
        team1 = row[TEAM0]
        team2 = row[TEAM1]
        wp = win_probability_heroes(team1, team2, mmr_dict, env)
        
        if row["winner"] == 0:
            crossentropy += np.log(wp)
        else:
            crossentropy += np.log(1-wp)
    
    return -crossentropy/len(replays)

if __name__ == "__main__":
    replays = read_replays("../data/processed/teams_20181001-20190123_processed.csv", "HeroLeague", "2.41")
    cut = int(len(replays)*0.8)
    replays_train = replays.iloc[:cut]
    replays_test = replays.iloc[cut:]
    
    
    param_grid = dict(mu = [25], mul_sigma = [1/3])
    
    for params in ParameterGrid(param_grid):
        mu = params["mu"]
        sigma = params["mul_sigma"]*mu
        print(mu, sigma)
        env = trueskill.setup(mu = params["mu"], sigma = sigma, beta = 1, tau = 0, draw_probability = 0)
    
        mmr_dict = calculate_mmr(replays_train, env)
        
        acc = accuracy(replays_test, mmr_dict)
            
        print("mu = {}, sigma = {}, accuracy: {}".format(mu, sigma, acc))
