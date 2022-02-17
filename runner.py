import trainer


lr = [1e-3]
batch_size = [100]
num_layers = [3]
epochs = 12
topk = 10
embedding_size = [32]
dropout = 0.0
generate_new_data = False
save_weights = True
show_graph = True
save_graphs = False
use_avg_rating = False




for l in lr:
    for b in batch_size:
        for e in embedding_size:
            for n in num_layers:
                trainer.Train(lr=l, dropout=dropout, batch_size=b, embedding_size=e, epochs=epochs, top_k=topk, num_layers=n,
                              save_weights=save_weights, show_graph=show_graph, save_graphs=save_graphs, use_avg_rating=use_avg_rating, generate_new_data=generate_new_data)

