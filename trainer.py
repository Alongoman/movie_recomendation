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
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
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
    sample_num = None

    parsed_data = np.array([elem[0].split("::")[:-1] for elem in ratings.values],dtype=int)

    ########## TODO remove this ########
    parsed_data = parsed_data[:sample_num]     #
    ####################################

    max_user = parsed_data[:,0].max()
    max_movie = parsed_data[:,1].max()

    users_set = set(parsed_data[:,0])
    movies_set = set(parsed_data[:,1])
    users_num = len(users_set)
    movies_num = len(movies_set)

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



    return train_data, train_data_for_acc, test_data, val_data, max_user, max_movie, user_movie_mat

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
    def __init__(self, features, movies_num, train_mat=None, num_neg=0):
        super(NCFData, self).__init__()
        """features is [user_id, movie_id, rating]
        if rating == 0 -> feature negative mean movies hasn't been watched
        """
        self.features_pos = features
        self.movies_num = movies_num
        self.train_mat = train_mat
        self.num_neg = num_neg
        self.labels = [0]*len(features)

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
        return user, movie ,rating ,label


def hit(gt_item, pred_items):
    if gt_item in pred_items:
        return 1
    return 0


def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index+2))
    return 0


def metrics(model, d_loader, top_k):
    HR, NDCG = [], []
    u_count = 0
    for user, movie, r, label in d_loader:
        u_count += 1
        print(f"test eval: {round(100*u_count/len(d_loader),2)}%")
        predictions = model(user, movie)
        _, indices = torch.topk(predictions, top_k)
        recommends = torch.take(movie, indices).cpu().numpy().tolist()

        print(f"user:{user[0]} | movie:{movie[0]} | rating:{r[0]}")

        pos_movie = movie[0].item() # first one is the actual watched movies from data
        HR.append(hit(pos_movie, recommends))
        NDCG.append(ndcg(pos_movie, recommends))

    return np.mean(HR), np.mean(NDCG)


class NCF_article(nn.Module):
    def __init__(self, user_num, item_num, factor_num, num_layers,
                 dropout, model, GMF_model=None, MLP_model=None):
        super(NCF, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors;
        num_layers: the number of layers in MLP model;
        dropout: dropout rate between fully connected layers;
        model: 'MLP', 'GMF', 'NeuMF-end', and 'NeuMF-pre';
        GMF_model: pre-trained GMF weights;
        MLP_model: pre-trained MLP weights.
        """
        self.dropout = dropout
        self.model = model
        self.GMF_model = GMF_model
        self.MLP_model = MLP_model
        self.name = "NCF_article_"+model

        self.embed_user_GMF = nn.Embedding(user_num, factor_num)
        self.embed_item_GMF = nn.Embedding(item_num, factor_num)
        self.embed_user_MLP = nn.Embedding(
            user_num, factor_num * (2 ** (num_layers - 1)))
        self.embed_item_MLP = nn.Embedding(
            item_num, factor_num * (2 ** (num_layers - 1)))

        MLP_modules = []
        for i in range(num_layers):
            input_size = factor_num * (2 ** (num_layers - i))
            MLP_modules.append(nn.Dropout(p=self.dropout))
            MLP_modules.append(nn.Linear(input_size, input_size//2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        if self.model in ['MLP', 'GMF']:
            predict_size = factor_num
        else:
            predict_size = factor_num * 2
        self.predict_layer = nn.Linear(predict_size, 1)

        self._init_weight_()

    def _init_weight_(self):
        """ We leave the weights initialization here. """
        if not self.model == 'NeuMF-pre':
            nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
            nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

            for m in self.MLP_layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
            nn.init.kaiming_uniform_(self.predict_layer.weight,
                                     a=1, nonlinearity='sigmoid')

            for m in self.modules():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
        else:
            # embedding layers
            self.embed_user_GMF.weight.data.copy_(
                self.GMF_model.embed_user_GMF.weight)
            self.embed_item_GMF.weight.data.copy_(
                self.GMF_model.embed_item_GMF.weight)
            self.embed_user_MLP.weight.data.copy_(
                self.MLP_model.embed_user_MLP.weight)
            self.embed_item_MLP.weight.data.copy_(
                self.MLP_model.embed_item_MLP.weight)

            # mlp layers
            for (m1, m2) in zip(
                    self.MLP_layers, self.MLP_model.MLP_layers):
                if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
                    m1.weight.data.copy_(m2.weight)
                    m1.bias.data.copy_(m2.bias)

            # predict layers
            predict_weight = torch.cat([
                self.GMF_model.predict_layer.weight,
                self.MLP_model.predict_layer.weight], dim=1)
            precit_bias = self.GMF_model.predict_layer.bias + \
                          self.MLP_model.predict_layer.bias

            self.predict_layer.weight.data.copy_(0.5 * predict_weight)
            self.predict_layer.bias.data.copy_(0.5 * precit_bias)

    def forward(self, user, item):
        if not self.model == 'MLP':
            embed_user_GMF = self.embed_user_GMF(user)
            embed_item_GMF = self.embed_item_GMF(item)
            output_GMF = embed_user_GMF * embed_item_GMF
        if not self.model == 'GMF':
            embed_user_MLP = self.embed_user_MLP(user)
            embed_item_MLP = self.embed_item_MLP(item)
            interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
            output_MLP = self.MLP_layers(interaction)

        if self.model == 'GMF':
            concat = output_GMF
        elif self.model == 'MLP':
            concat = output_MLP
        else:
            concat = torch.cat((output_GMF, output_MLP), -1)

        prediction = self.predict_layer(concat)
        return prediction.view(-1)


class NCF(nn.Module):
    '''description'''
    def __init__(self, user_count, movie_count, embedding_size=32, hidden_layers=(64,32,16,8), dropout_rate=None, output_range=(0,1)):
        super().__init__()
        self.name = "NCF"
        self.user_hash_size = user_count
        self.movie_hash_size = movie_count
        self.user_embbeding = nn.Embedding(num_embeddings=user_count, embedding_dim=embedding_size)
        self.movie_embbeding = nn.Embedding(num_embeddings=movie_count, embedding_dim=embedding_size)
        self.mlp = self._gen_MLP(embedding_size, hidden_layers, dropout_rate)
        self.sigmoid = nn.Sigmoid()

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

        for num_units in hidden_layers_units[1:]:
            hidden_layers.append(nn.Linear(in_features=input_units, out_features=num_units))
            hidden_layers.append(nn.ReLU())
            if dropout_rate:
                hidden_layers.append(nn.Dropout(dropout_rate))
            input_units = num_units

        hidden_layers.append(nn.Linear(in_features=hidden_layers_units[-1], out_features=1))
        # hidden_layers.append(nn.Sigmoid())

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

    def forward(self, user_id, movie_id):
        user_features = self.user_embbeding((user_id %  self.user_hash_size).long())
        movie_features = self.movie_embbeding((movie_id % self.movie_hash_size).long())
        x = torch.cat([user_features, movie_features], dim=1)
        if hasattr(self,'dropout'):
            x = self.dropout(x)
        x = self.mlp(x)
        return x.view(-1)


def Train(lr=1e-3, dropout=0.0, batch_size=256, epochs=1, top_k=10, embedding_size=32, num_layers=3,num_neg=3,test_num_neg=99, save_weights=True, show_graph=True, generate_new_data=False, recomendation_num=0):

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
        train_data, train_data_for_acc, test_data, val_data, max_user, max_movie, user_movie_mat = load_data(ratings)
        save(train_data, f"{data_save_path}train__thresh_{label_threshold}.pkl")
        save(train_data_for_acc, f"{data_save_path}train_for_acc__thresh_{label_threshold}.pkl")
        save(test_data, f"{data_save_path}test__thresh_{label_threshold}.pkl")
        save(val_data, f"{data_save_path}val__thresh_{label_threshold}.pkl")

    else:
        #load data
        parsed_data = np.array([elem[0].split("::")[:-1] for elem in ratings.values],dtype=int)
        max_user = parsed_data[:,0].max() + 1
        max_movie = parsed_data[:,1].max() + 1
        users_set = set(parsed_data[:,0])
        movies_set = set(parsed_data[:,1])
        sample_num = "all"
        train_data = load(f"{data_save_path}train_samples_{sample_num}.pkl")
        train_data_for_acc = load(f"{data_save_path}train_for_acc_samples_{sample_num}.pkl")
        test_data = load(f"{data_save_path}test_samples_{sample_num}.pkl")
        val_data = load(f"{data_save_path}val_samples_{sample_num}.pkl")
        user_movie_mat = get_user_movie_mat(train_data,max_user,max_movie)


    # construct the train and test datasets
    train_dataset = NCFData(train_data, max_movie, user_movie_mat, num_neg)
    test_dataset = NCFData(test_data, max_movie, user_movie_mat,test_num_neg)
    val_dataset = NCFData(val_data, max_movie, user_movie_mat,test_num_neg)
    train_acc_dataset = NCFData(train_data_for_acc, max_movie, user_movie_mat,test_num_neg)

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = data.DataLoader(test_dataset,	batch_size=test_num_neg+1, shuffle=False, num_workers=0)
    val_loader = data.DataLoader(val_dataset, batch_size=test_num_neg+1, shuffle=False, num_workers=0)
    train_acc_loader = data.DataLoader(train_acc_dataset, batch_size=test_num_neg+1, shuffle=False, num_workers=0)


    model = NCF(user_count=max_user, movie_count=max_movie, embedding_size=embedding_size, hidden_layers=hidden_layers, dropout_rate=dropout)
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
    for epoch in range(epochs):
        model.train() # Enable dropout (if have).
        start_time = time.time()
        train_loader.dataset.neg_sample() # add unrecomended movies to dataset
        u_count = 0
        avg_loss = 0
        for user, movie, r, label in train_loader:
            u_count += 1
            print(f"epoch {epoch}/{epochs} |train: {round(100*u_count/len(train_loader),2)}% | hr_test: {hr_val_list} | hr_train: {hr_train_list} | avg loss: {np.mean(losses)}")
            user = user.cpu()
            movie = movie.cpu()
            label = label.float().cpu()
            model.zero_grad()
            prediction = model(user, movie)
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


    print(f"total time elapsed is: {time.strftime('%H: %M: %S', time.gmtime(time.time()-t0))}")

    plt.figure()
    plt.plot(losses)
    plt.title(f"{model.name} loss curve")
    plt.suptitle(f"lr: {lr} | batch size: {batch_size} | layers: {hidden_layers}")
    plt.xlabel("iters")
    file_name = os.path.join(main_path,f"results/{model.name}__loss__lr_{lr}__batch_{batch_size}__layers_{hidden_layers}.png") 
    plt.savefig(file_name)

    plt.figure()
    plt.plot(hr_val_list, label='val acc')
    plt.plot(hr_train_list, label='train acc')
    plt.title(f"{model.name} HR@10 accuracy | best val acc={round(best_hr,3)}")
    plt.suptitle(f"lr: {lr} | batch size: {batch_size} | layers: {hidden_layers}")
    plt.xlabel("epoch")
    plt.legend()
    file_name = os.path.join(main_path,f"results/{model.name}__hr10_acc_{round(best_hr,3)}__lr_{lr}__batch_{batch_size}__layers_{hidden_layers}.png") 
    plt.savefig(file_name)

    plt.figure()
    plt.plot(ndgc_val_list, label='val acc')
    plt.plot(ndgc_train_list, label='train acc')
    plt.title(f"{model.name} NDGC accuracy | best val acc={round(best_ndcg,3)}")
    plt.suptitle(f"lr: {lr} | batch size: {batch_size} | layers: {hidden_layers}")
    plt.xlabel("epoch")
    plt.legend()
    file_name = os.path.join(main_path,f"results/{model.name}__ndgc_acc_{round(best_ndcg,3)}__lr_{lr}__batch_{batch_size}__layers_{hidden_layers}.png") 
    plt.savefig(file_name)

    if show_graph:
        plt.show()

    if recomendation_num:
        get_recomendation(model=model,movies_data=movies_data, max_user=max_user, max_movie=max_movie, recomendation_num=recomendation_num)


def get_recomendation(model,movies_data, max_user, max_movie, recomendation_num=10):
    user_tensor = torch.Tensor([max_user-1]*max_movie)
    movie_tensor = torch.Tensor([m for m in range(max_movie)])
    predictions = model(user_tensor,movie_tensor)
    _, indices = torch.topk(predictions, recomendation_num)
    recommends = torch.take(movie_tensor, indices).cpu().numpy().tolist()
    for r in recommends:
        movie_name = movies_data["data"][int(r)-1].split("::")[1]
        movie_genre = movies_data["data"][int(r)-1].split("::")[2]
        print(f"{movie_name} :: {movie_genre}")


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
    gpu = "0"
    label_threshold = 0
    generate_new_data = args.generate_data
    recomendation_num = args.recomendation_num

    Train(lr=lr, dropout=dropout, batch_size=batch_size, epochs=epochs, top_k=top_k, embedding_size=embedding_size,
          num_layers=num_layers, num_neg=num_neg, test_num_neg=test_num_neg, save_weights=save_weights,
          show_graph=show_graph, generate_new_data=generate_new_data, recomendation_num=recomendation_num)

