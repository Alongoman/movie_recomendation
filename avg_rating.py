import numpy as np
import pandas as pd
import pickle
import os

def save(data, file):
    with open(file, 'wb') as outp:
        pickle.dump(data, outp, pickle.HIGHEST_PROTOCOL)

def load(file):
    with open(file, 'rb') as inp:
        return pickle.load(inp)

gen = True


label_threshold = 0

main_path = os.getcwd()

ratings_path = f"{main_path}/ml-1m/ratings.dat"
ratings = pd.read_csv(ratings_path)

movies_path = f"{main_path}/ml-1m/movies.dat"
movies_data = pd.read_csv(movies_path,names=["data"],encoding="latin-1", sep="\t")


data_save_path = os.path.join(main_path,'post_process/')
print(f"save file path: {data_save_path}")

if gen:
    parsed_data = np.array([elem[0].split("::")[:-1] for elem in ratings.values],dtype=int)


    avg_movie_rating = {}
    total_movie_rated = {}

    for movie_id in range(1,3953):
        total_movie_rated[movie_id] = 0
        avg_movie_rating[movie_id] = 0


    for sample in parsed_data: # user interaction mat for explicit data (similar to chpater 2.1 in NCF article only taking rating into account)
        movie_id = sample[1]
        rating = sample[2]
        total_movie_rated[movie_id] += 1
        avg_movie_rating[movie_id] += rating

    for movie_id,counts in total_movie_rated.items():
        if counts <= 0:
            counts = 1
        avg_movie_rating[movie_id] = avg_movie_rating[movie_id]/counts


    save(avg_movie_rating, f"{data_save_path}avg_movie_rating_thresh_{label_threshold}.pkl")

else:
    avg_movie_rating = load(f"{data_save_path}avg_movie_rating_thresh_{label_threshold}.pkl")

for k,v in avg_movie_rating.items():
    print(f"{k} :: {v}")