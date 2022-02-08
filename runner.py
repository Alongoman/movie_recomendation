from movie_recomendation import trainer


lr = [5e-3,1e-3,5e-4]
batch_size = [100,200,300]
epochs = 1
topk = 10
embedding_size = [16,32,64]
num_layers = [3,4,5]
save_weights = True
show_graph = False

for l in lr:
    for b in batch_size:
        for e in embedding_size:
            for n in num_layers:
                trainer.Train(lr=l, batch_size=b, embedding_size=e, epochs=epochs, top_k=topk, num_layers=n,
                              save_weights=save_weights, show_graph=show_graph)

