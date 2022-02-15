import trainer


lr = [5e-3,1e-3,5e-4]
batch_size = [100]
epochs = 30
topk = 10
embedding_size = [32]
num_layers = [3,4]
generate_new_data = False
save_weights = True
show_graph = False
save_graphs = True
use_avg_rating = True




for l in lr:
    for b in batch_size:
        for e in embedding_size:
            for n in num_layers:
                trainer.Train(lr=l, batch_size=b, embedding_size=e, epochs=epochs, top_k=topk, num_layers=n,
                              save_weights=save_weights, show_graph=show_graph, save_graphs=save_graphs, use_avg_rating=use_avg_rating, generate_new_data=generate_new_data)

