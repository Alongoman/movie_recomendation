import time
import numpy as np
import matplotlib.pyplot as plt
import collections
import random
import pandas as pd
import scipy.sparse as sp
import pickle
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import os



def get_time_remained(time_remained):
    hours = str(time_remained//3600)
    minutes = str((time_remained%3600)//60)
    seconds = str(time_remained%60)
    if len(seconds) == 1:
        seconds = "0"+seconds
    if len(minutes) == 1:
        minutes = "0"+minutes
    if len(hours) == 1:
        hours = "0"+hours

    return hours,minutes,seconds


def load_data(ratings, movies_count_for_test=1, good_movie_tresh=0):
    ''' loading data from the CSV file'''
    parsed_data = np.array([elem[0].split("::")[:-1] for elem in ratings.values],dtype=int)

    max_user = parsed_data[:,0].max()
    max_movie = parsed_data[:,1].max()

    users_set = set(parsed_data[:,0])
    movies_set = set(parsed_data[:,1])
    users_num = len(users_set)
    movies_num = len(movies_set)
    avg_movie_rating = {}
    total_movie_rated = {}

    for movie_id in range(1,max_movie+1):
        total_movie_rated[movie_id] = 0
        avg_movie_rating[movie_id] = 0

    movies_dict = collections.Counter(parsed_data[:,1])
    # load ratings as a dok matrix
    user_movie_mat = sp.dok_matrix((max_user+1, max_movie+1, ), dtype=np.float32)
    user_movie_mat[0,:] = -1
    user_movie_mat[:,0] = -1
    for sample in parsed_data: # user interaction mat for explicit data (similar to chpater 2.1 in NCF article only taking rating into account)
        user_id = sample[0]
        movie_id = sample[1]
        rating = sample[2]
        user_movie_mat[user_id,movie_id] = rating
        total_movie_rated[movie_id] += 1
        avg_movie_rating[movie_id] += rating

    val_data = []
    test_data = []
    train_data_for_acc = []
    train_data = []
    t0 = time.time()
    # take high rated movie and add him 99 not rated movies to get val/test set
    prev_user = parsed_data[0][0]
    i=-1
    for u_id in users_set:
        i+=1
        j=-1
        movies_for_val  = []
        movies_for_test = []
        movies_for_train_acc = []
        movie_list = list(movies_set)
        random.shuffle(movie_list)
        for m_id in movie_list:
            j+=1
            t1 = time.time()
            done_presentage = round(100*(j + i*movies_num)/(users_num*movies_num*2),2)
            time_remained = int(((t1-t0)/(1e-2+ done_presentage))*(100-done_presentage))
            hours,minutes,seconds = get_time_remained(time_remained)
            print(f"ETA {hours}:{minutes}:{seconds} (h:m:s) | {done_presentage}% a")
            if movies_dict[m_id] <=1:
                continue
            rating = user_movie_mat[u_id,m_id]
            if rating > good_movie_tresh:
                if len(movies_for_val) < movies_count_for_test:
                    movies_for_val.append([u_id,m_id,rating])
                    user_movie_mat[u_id,m_id] = 0
                    movies_dict[m_id] -= 1
                elif len(movies_for_test) < movies_count_for_test:
                    movies_for_test.append([u_id,m_id,rating])
                    user_movie_mat[u_id,m_id] = 0
                    movies_dict[m_id] -= 1
                elif len(movies_for_train_acc) < movies_count_for_test:
                    movies_for_train_acc.append([u_id,m_id,rating])
                else:
                    break
        val_data += movies_for_val
        test_data += movies_for_test
        train_data_for_acc += movies_for_train_acc

    i=-1
    for u_id in users_set:
        i+=1
        j=-1
        for m_id in movies_set:
            j+=1
            t1 = time.time()
            done_presentage = round(50 + 100*(j + i*movies_num)/(users_num*movies_num*2),2)
            time_remained = int(((t1-t0)/done_presentage)*(100-done_presentage))
            hours,minutes,seconds = get_time_remained(time_remained)
            print(f"ETA {hours}:{minutes}:{seconds} (h:m:s) | {done_presentage}% b")
            rating = user_movie_mat[u_id,m_id]
            if rating > good_movie_tresh:
                train_data.append([u_id,m_id,rating])

    for movie_id,counts in total_movie_rated.items():
        avg_movie_rating[movie_id] = avg_movie_rating[movie_id]/counts

    return train_data, train_data_for_acc, test_data, val_data, max_user, max_movie, avg_movie_rating, user_movie_mat, good_movie_tresh


def save(data, file):
    with open(file, 'wb') as outp:
        pickle.dump(data, outp, pickle.HIGHEST_PROTOCOL)


def load(file):
    with open(file, 'rb') as inp:
        return pickle.load(inp)


def get_user_movie_mat(train_data,max_user,max_movie):
    user_movie_mat = sp.dok_matrix((max_user, max_movie, ), dtype=np.float32)
    user_movie_mat[0,:] = -1
    user_movie_mat[:,0] = -1
    for sample in train_data:
        user,movie,rating = sample[0], sample[1], sample[2]
        user_movie_mat[user,movie] = rating
    return user_movie_mat


class NCFData(data.Dataset):
    def __init__(self, features, movies_num, train_mat=None, num_neg=0, avg_movie_rating={}):
        super(NCFData, self).__init__()
        """features is [user_id, movie_id, rating]
        if rating == 0 -> feature negative mean movies hasn't been watched
        """
        self.features_pos = features
        self.movies_num = movies_num
        self.train_mat = train_mat
        self.num_neg = num_neg
        self.labels = [0]*len(features)
        if not avg_movie_rating:
            for i in range(movies_num):
                avg_movie_rating[i] = 0
        self.avg_movie_rating = avg_movie_rating

    def neg_sample(self):
        print("sampling unrecommended movies for dataset...")
        self.features_fill = []
        self.labels_fill = []
        for x in self.features_pos:
            u = x[0]
            self.features_fill.append(x.copy())
            self.labels_fill.append(1)
            for t in range(self.num_neg):
                j = np.random.randint(self.movies_num)
                while (u, j) in self.train_mat:
                    j = np.random.randint(self.movies_num)
                self.features_fill.append([u, j, 0])
                self.labels_fill.append(0)

    def __len__(self):
        return (self.num_neg + 1) * len(self.labels)

    def __getitem__(self, idx):
        features = self.features_fill
        labels = self.labels_fill
        if not self.features_fill:
            print("did not sampled negative movies, using only positive ones")
            features = self.features_pos
            labels = self.labels
        user = features[idx][0]
        movie = features[idx][1]
        rating = features[idx][2]
        label = labels[idx]
        return user, movie ,rating ,label, self.avg_movie_rating[movie]


def hit(item, pred_items):
    if item in pred_items:
        return 1
    return 0


def ndcg(item, pred_items):
    if item in pred_items:
        index = pred_items.index(item)
        return np.reciprocal(np.log2(index+2))
    return 0


def metrics(model, d_loader, top_k):
    HR, NDCG = [], []
    u_count = 0
    for user, movie, r, label, avg_r in d_loader:
        u_count += 1
        print(f"test eval: {round(100*u_count/len(d_loader),2)}%")
        predictions = model(user, movie, avg_r.float())
        _, indices = torch.topk(predictions, top_k)
        recommends = torch.take(movie, indices).cpu().numpy().tolist()

        print(f"user:{user[0]} | movie:{movie[0]} | rating:{r[0]}")

        pos_movie = movie[0].item() # first one is the actual watched movies from data
        HR.append(hit(pos_movie, recommends))
        NDCG.append(ndcg(pos_movie, recommends))

    return np.mean(HR), np.mean(NDCG)


class NCF(nn.Module):
    '''description'''
    def __init__(self, user_count, movie_count, embedding_size=32, hidden_layers=(64,32,16,8), dropout_rate=None, output_range=(0,1), use_avg_rating=False):
        super().__init__()
        self.name = "NCF"
        if use_avg_rating:
            self.name += "_explicit"
        self.use_avg_rating = use_avg_rating
        self.user_hash_size = user_count
        self.movie_hash_size = movie_count
        self.user_embbeding = nn.Embedding(num_embeddings=user_count, embedding_dim=embedding_size)
        self.movie_embbeding = nn.Embedding(num_embeddings=movie_count, embedding_dim=embedding_size)
        self.sigmoid = nn.Sigmoid()
        self.mlp = self._gen_MLP(embedding_size, hidden_layers, dropout_rate)

        if dropout_rate:
            self.dropout = nn.Dropout(dropout_rate)

        assert output_range and len(output_range) == 2, "output range has to be a tuple with 2 integers"
        self.norm_min = min(output_range)
        self.norm_range = abs(output_range[0]-output_range[1])+1
        self._init_params()


    def _gen_MLP(self, embedding_size, hidden_layers_units, dropout_rate):
        ''' generate the multi layer preceptor'''
        assert (embedding_size*2) == hidden_layers_units[0], "first input layer number hidden must be equal to twice the embbeding size"

        hidden_layers = []
        input_units = hidden_layers_units[0]

        if self.use_avg_rating:
            input_units += 1

        for num_units in hidden_layers_units[1:]:
            hidden_layers.append(nn.Linear(in_features=input_units, out_features=num_units))
            hidden_layers.append(nn.ReLU())
            if dropout_rate:
                hidden_layers.append(nn.Dropout(dropout_rate))
            input_units = num_units

        hidden_layers.append(nn.Linear(in_features=hidden_layers_units[-1], out_features=1))
        # hidden_layers.append(nn.Sigmoid()) # if using BCElosswithlogits sigmoid is already included there

        return nn.Sequential(*hidden_layers)


    def _init_params(self):
        ''' initialize the model params'''
        def weights_init(m):
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.user_embbeding.weight.data.uniform_(-0.05,0.05)
        self.movie_embbeding.weight.data.uniform_(-0.05,0.05)
        self.mlp.apply(weights_init)

    def forward(self, user_id, movie_id, avg_rating=0):
        user_features = self.user_embbeding((user_id % self.user_hash_size).long())
        movie_features = self.movie_embbeding((movie_id % self.movie_hash_size).long())

        if self.use_avg_rating:
            x = torch.cat([user_features, movie_features, torch.reshape(avg_rating,(avg_rating.shape[0],1))], dim=1)
        else:
            x = torch.cat([user_features, movie_features], dim=1)

        if hasattr(self,'dropout'):
            x = self.dropout(x)
        x = self.mlp(x)
        return x.view(-1)


def Train(lr=1e-3, dropout=0.0, batch_size=256, epochs=1, top_k=10, embedding_size=32, num_layers=3,num_neg=3,test_num_neg=99, save_weights=True,
          show_graph=True, save_graphs=True, generate_new_data=False, use_avg_rating=False, recomendation_num=0, user_movies=[]):

    recomendation_mode = recomendation_num and (len(user_movies) > 2)

    hidden = []
    for i in range(num_layers):
        s = (2*embedding_size)/(2**i)
        if s < 2 :
            break
        hidden.append(int(s))
    hidden_layers = tuple(hidden)

    t0 = time.time()

    main_path = os.getcwd()

    ratings_path = f"{main_path}/ml-1m/ratings.dat"
    ratings = pd.read_csv(ratings_path)

    movies_path = f"{main_path}/ml-1m/movies.dat"
    movies_data = pd.read_csv(movies_path,names=["data"],encoding="latin-1", sep="\t")


    data_save_path = os.path.join(main_path,'post_process/')
    print(f"save file path: {data_save_path}")

    if generate_new_data:
        train_data, train_data_for_acc, test_data, val_data, max_user, max_movie, avg_movie_rating, user_movie_mat, label_threshold = load_data(ratings)
        save(train_data, f"{data_save_path}train_thresh_{label_threshold}.pkl")
        save(train_data_for_acc, f"{data_save_path}train_for_acc_thresh_{label_threshold}.pkl")
        save(test_data, f"{data_save_path}test_thresh_{label_threshold}.pkl")
        save(val_data, f"{data_save_path}val_thresh_{label_threshold}.pkl")
        save(avg_movie_rating, f"{data_save_path}avg_movie_rating_thresh_{label_threshold}.pkl")

    else:
        #load data
        parsed_data = np.array([elem[0].split("::")[:-1] for elem in ratings.values],dtype=int)
        max_user = parsed_data[:,0].max() + 1
        max_movie = parsed_data[:,1].max() + 1
        users_set = set(parsed_data[:,0])
        movies_set = set(parsed_data[:,1])
        label_threshold = 0
        train_data = load(f"{data_save_path}train_thresh_{label_threshold}.pkl")
        train_data_for_acc = load(f"{data_save_path}train_for_acc_thresh_{label_threshold}.pkl")
        test_data = load(f"{data_save_path}test_thresh_{label_threshold}.pkl")
        val_data = load(f"{data_save_path}val_thresh_{label_threshold}.pkl")
        avg_movie_rating = load(f"{data_save_path}avg_movie_rating_thresh_{label_threshold}.pkl")
        if not(recomendation_mode):
            user_movie_mat = get_user_movie_mat(train_data,max_user,max_movie)

    if recomendation_mode:
        for id,m in user_movies:
            train_data.append([max_user, id, 5.0])
        val_data.append(train_data.pop())
        test_data.append(train_data.pop())
        max_user += 1
        user_movie_mat = get_user_movie_mat(train_data,max_user,max_movie)

    # construct the train and test datasets
    train_dataset = NCFData(train_data, max_movie, user_movie_mat, num_neg, avg_movie_rating=avg_movie_rating)
    test_dataset = NCFData(test_data, max_movie, user_movie_mat,test_num_neg, avg_movie_rating=avg_movie_rating)
    val_dataset = NCFData(val_data, max_movie, user_movie_mat,test_num_neg, avg_movie_rating=avg_movie_rating)
    train_acc_dataset = NCFData(train_data_for_acc, max_movie, user_movie_mat,test_num_neg, avg_movie_rating=avg_movie_rating)

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = data.DataLoader(test_dataset,	batch_size=test_num_neg+1, shuffle=False, num_workers=0)
    val_loader = data.DataLoader(val_dataset, batch_size=test_num_neg+1, shuffle=False, num_workers=0)
    train_acc_loader = data.DataLoader(train_acc_dataset, batch_size=test_num_neg+1, shuffle=False, num_workers=0)

    model = NCF(user_count=max_user, movie_count=max_movie, embedding_size=embedding_size, hidden_layers=hidden_layers, dropout_rate=dropout, use_avg_rating=use_avg_rating)
    print(f"model name: {model.name}")

    val_loader.dataset.neg_sample()
    train_acc_loader.dataset.neg_sample()


    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    ########################### TRAINING #####################################
    count, best_hr = 0, 0
    hr_train_list, hr_val_list = [],[]
    ndgc_train_list, ndgc_val_list = [],[]
    losses = []
    for epoch in range(1,epochs+1):
        model.train() # Enable dropout (if have).
        start_time = time.time()
        train_loader.dataset.neg_sample() # add unrecomended movies to dataset
        u_count = 0
        avg_loss = 0
        for user, movie, r, label, avg_r in train_loader:
            u_count += 1
            print(f"epoch {epoch}/{epochs} |train: {round(100*u_count/len(train_loader),2)}% | hr_test: {hr_val_list} | hr_train: {hr_train_list} | avg loss: {np.mean(losses)}")
            user = user.cpu()
            movie = movie.cpu()
            avg_r = avg_r.float().cpu()
            label = label.float().cpu()
            model.zero_grad()
            prediction = model(user, movie, avg_r)
            loss = loss_function(prediction, label)
            loss.backward()
            optimizer.step()
            # writer.add_scalar('data/loss', loss.movie(), count)
            count += 1
            avg_loss += loss.item()

        losses.append(avg_loss/count)
        model.eval()
        HR_val, NDCG_val = metrics(model, val_loader, top_k)
        HR_train, NDCG_train = metrics(model, train_acc_loader, top_k)
        hr_val_list.append(HR_val)
        hr_train_list.append(HR_train)
        ndgc_val_list.append(NDCG_val)
        ndgc_train_list.append(NDCG_train)
        elapsed_time = time.time() - start_time
        print("The time elapse of epoch {:03d}".format(epoch) + " is: " +
              time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
        print("HR val: {:.3f}\tNDCG: {:.3f}".format(HR_val, NDCG_val))
        print("HR train: {:.3f}\tNDCG: {:.3f}".format(HR_train, NDCG_train))

        if HR_val > best_hr:
            best_hr, best_ndcg, best_epoch = HR_val, NDCG_val, epoch
            # if save_weights:
            # if not os.path.exists(model_path):
            # 	os.mkdir(model_path)
            # torch.save(model,
            # 	'{}{}.pth'.format(model_path, model))

    print("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(
        best_epoch, best_hr, best_ndcg))


    test_loader.dataset.neg_sample()
    model.eval()
    HR_test, NDCG_test = metrics(model, test_loader, top_k)

    print(f"total time elapsed is: {time.strftime('%H: %M: %S', time.gmtime(time.time()-t0))}")
    print("___________________________ test accuracy ___________________________")
    print(f"test set HR@{top_k} accuracy: {round(100*HR_test,5)}%")
    print(f"test set NDCG@{top_k} accuracy: {round(100*NDCG_test,5)}%")
    print("_____________________________________________________________________")

    plt.figure()
    plt.plot(losses)
    plt.title(f"{model.name} loss curve")
    plt.suptitle(f"lr: {lr} | batch size: {batch_size} | layers: {hidden_layers}")
    plt.xlabel("iters")
    if save_graphs:
        file_name = os.path.join(main_path,f"results/{model.name}__loss__lr_{lr}__batch_{batch_size}__layers_{hidden_layers}.png")
        plt.savefig(file_name)

    plt.figure()
    plt.plot(hr_val_list, label='val acc')
    plt.plot(hr_train_list, label='train acc')
    plt.title(f"{model.name} HR@10 accuracy | best val acc={round(best_hr,4)}")
    plt.suptitle(f"lr: {lr} | batch size: {batch_size} | layers: {hidden_layers}")
    plt.xlabel("epoch")
    plt.legend()
    if save_graphs:
        file_name = os.path.join(main_path,f"results/{model.name}__hr10_acc_{round(best_hr,4)}__lr_{lr}__batch_{batch_size}__layers_{hidden_layers}.png")
        plt.savefig(file_name)

    plt.figure()
    plt.plot(ndgc_val_list, label='val acc')
    plt.plot(ndgc_train_list, label='train acc')
    plt.title(f"{model.name} NDGC accuracy | best val acc={round(best_ndcg,4)}")
    plt.suptitle(f"lr: {lr} | batch size: {batch_size} | layers: {hidden_layers}")
    plt.xlabel("epoch")
    plt.legend()
    if save_graphs:
        file_name = os.path.join(main_path,f"results/{model.name}__ndgc_acc_{round(best_ndcg,4)}__lr_{lr}__batch_{batch_size}__layers_{hidden_layers}.png")
        plt.savefig(file_name)

    if show_graph:
        plt.show()

    if recomendation_mode:
        get_recomendation(model=model,movies_data=movies_data, max_user=max_user, max_movie=max_movie, recomendation_num=recomendation_num, epochs=epoch, hr_acc=best_hr)


def get_recomendation(model,movies_data, max_user, max_movie, recomendation_num=10, epochs=0, hr_acc=0):
    user_tensor = torch.Tensor([max_user-1]*max_movie)
    movie_tensor = torch.Tensor([m for m in range(max_movie)])
    predictions = model(user_tensor,movie_tensor)
    _, indices = torch.topk(predictions, recomendation_num)
    recommends = torch.take(movie_tensor, indices).cpu().numpy().tolist()

    print("################################################")
    print()
    print(f"here are your recomendations after {epochs} epochs, expected to have {round(100*hr_acc,2)}% to get at least 1 good recomendations in the first 10 movies")
    print()
    for r in recommends:
        movie_name = movies_data["data"][int(r)-1].split("::")[1]
        movie_genre = movies_data["data"][int(r)-1].split("::")[2]
        print(f"{movie_name} :: {movie_genre}")
    print()
    print("you should try some of the above !")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--learning_rate', default=1e-3,help="learning rate")
    parser.add_argument('--dropout', default=0.0,help="dropout rate")
    parser.add_argument('-b', '--batch_size', default=256,help="batch size")
    parser.add_argument('-e', '--epochs', default=1,help="how many epochs to run")
    parser.add_argument('-k', '--topk', default=10,help="tok k number, for HR@k")
    parser.add_argument('-emb', '--embedding_size', default=32,help="dimension of the embedding layer")
    parser.add_argument('--layers', default=3,help="layers number, max is 1 + log2(embedding_size)")
    parser.add_argument('--num_neg', default=3,help="for each recommended movie will add N unrecomended movies to the train set")
    parser.add_argument('--test_num_neg', default=99,help="for each recommended movie will add N unrecomended movies to the validation and test set")
    parser.add_argument('--save_weights', action='store_true',help="save best weights")
    parser.add_argument('--show_graph', action='store_true',help="print graphs, the program will not end until all graphs are closed")
    parser.add_argument('--generate_data', action='store_true',help="generate all the data from scratch, long process")
    parser.add_argument('--recomendation_num', default=0,help="change this value in order to get more/less recomendations")
    args = parser.parse_args()

    lr = args.learning_rate
    dropout = args.dropout
    batch_size = args.batch_size
    epochs = args.epochs
    top_k = args.topk
    embedding_size = args.embedding_size
    num_layers = args.layers
    num_neg = args.num_neg
    test_num_neg = args.test_num_neg
    save_weights = args.save_weights
    show_graph = args.show_graph
    label_threshold = 0
    generate_new_data = args.generate_data
    recomendation_num = args.recomendation_num

    Train(lr=lr, dropout=dropout, batch_size=batch_size, epochs=epochs, top_k=top_k, embedding_size=embedding_size,
          num_layers=num_layers, num_neg=num_neg, test_num_neg=test_num_neg, save_weights=save_weights,
          show_graph=show_graph, generate_new_data=generate_new_data, recomendation_num=recomendation_num)

