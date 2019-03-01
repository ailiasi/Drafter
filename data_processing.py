import numpy as np
import pandas as pd

# TODO: Use the Enum class to encode heroes, maps and modes

HEROES = {'Abathur': 0, 'Alarak': 1, 'Alexstrasza': 2, 'Ana': 3, "Anub'arak": 4,
          'Artanis': 5, 'Arthas': 6, 'Auriel': 7, 'Azmodan': 8, 'Blaze': 9,
          'Brightwing': 10, 'Cassia': 11, 'Chen': 12, 'Cho': 13, 'Chromie': 14,
          'D.Va': 15, 'Deckard': 16, 'Dehaka': 17, 'Diablo': 18, 'E.T.C.': 19,
          'Falstad': 20, 'Fenix': 21, 'Gall': 22, 'Garrosh': 23, 'Gazlowe': 24,
          'Genji': 25, 'Greymane': 26, "Gul'dan": 27, 'Hanzo': 28, 'Illidan': 29,
          'Imperius': 30, 'Jaina': 31, 'Johanna': 32, 'Junkrat': 33, "Kael'thas": 34,
          "Kel'Thuzad": 35, 'Kerrigan': 36, 'Kharazim': 37, 'Leoric': 38, 'Li Li': 39,
          'Li-Ming': 40, 'Lt. Morales': 41, 'Lunara': 42, 'Lúcio': 43, 'Maiev': 44,
          "Mal'Ganis": 45, 'Malfurion': 46, 'Malthael': 47, 'Medivh': 48, 'Mephisto': 49,
          'Muradin': 50, 'Murky': 51, 'Nazeebo': 52, 'Nova': 53, 'Orphea': 54,
          'Probius': 55, 'Ragnaros': 56, 'Raynor': 57, 'Rehgar': 58, 'Rexxar': 59,
          'Samuro': 60, 'Sgt. Hammer': 61, 'Sonya': 62, 'Stitches': 63, 'Stukov': 64,
          'Sylvanas': 65, 'Tassadar': 66, 'The Butcher': 67, 'The Lost Vikings': 68, 'Thrall': 69,
          'Tracer': 70, 'Tychus': 71, 'Tyrael': 72, 'Tyrande': 73, 'Uther': 74,
          'Valeera': 75, 'Valla': 76, 'Varian': 77, 'Whitemane': 78, 'Xul': 79,
          'Yrel': 80, 'Zagara': 81, 'Zarya': 82, 'Zeratul': 83, "Zul'jin": 84}

MAPS = {'Alterac Pass': 0, 
        'Battlefield of Eternity': 1, 
        "Blackheart's Bay": 2, 
        'Braxis Holdout': 3, 
        'Cursed Hollow': 4,
        'Dragon Shire': 5, 
        'Garden of Terror': 6, 
        'Hanamura Temple': 7, 
        'Infernal Shrines': 8, 
        'Sky Temple': 9,
        'Tomb of the Spider Queen': 10, 
        'Towers of Doom': 11, 
        'Volskaya Foundry': 12, 
        'Warhead Junction': 13}

MODES = {'HeroLeague': 0, 'QuickMatch': 1, 'TeamLeague': 2, 'UnrankedDraft':3}

HEROCOLUMNS = ["hero" + str(i) for i in range(1,11)]
TEAM0 = HEROCOLUMNS[:5]
TEAM1 = HEROCOLUMNS[5:]

def collect_heroes_per_replay(df, hero_field, grouping_fields, team_fields):
    # TODO: Make sure that the players are in correct order
    df_new = pd.DataFrame()
    groups = df.groupby(grouping_fields)
    df_new = groups.agg({hero_field: lambda x: tuple(x)})
    df_new[["hero" + str(i) for i in range(1,11)]] = df_new[hero_field].apply(pd.Series)
    df_new = df_new.drop(hero_field, axis = 1)
    df_new["winner"] = groups[team_fields].apply(lambda df: df[team_fields[0]][df[team_fields[1]]==True].iloc[0])
    df_new = df_new.reset_index()
    return df_new

def encode_row(row):
    fields = [HEROES[hero] for hero in row[TEAM0]] + \
             [HEROES[hero] + 130 for hero in row[TEAM1]] + \
             [MAPS[row["game_map"]] + 100, 
              MAPS[row["game_map"]] + 100 + 130,
              MODES[row["game_type"]] + 120,
              MODES[row["game_type"]] + 120 + 130,
              260 + row["winner"]]
    return pd.Series(fields, index = TEAM0 + TEAM1 + ["map0", "map1", "mode0", "mode1", "winner"])

def binary_encode(row):
    encode = np.zeros(262)
    encode[row] = 1
    return pd.Series(encode)

def read_replays(filename, game_type, game_version):
    replays = pd.read_csv(filename, parse_dates = ["game_date"])
    replays = (replays[(replays["game_type"] == game_type) & (replays["game_version"].str.startswith(game_version))]
                   .sort_values("game_date")
                   .dropna())
    return replays


if __name__ == "__main__":
    df = pd.read_csv("data/teams_patch_2.42.0.71449.csv", nrows = 20)
    print(df)
    grouping_fields = ["id", "game_date", "game_type", "game_map", "region"]
    df_new = collect_heroes_per_replay(df, 
                                       "players_hero", 
                                       grouping_fields, 
                                       ["players_team", "players_winner"])
    print(df_new)