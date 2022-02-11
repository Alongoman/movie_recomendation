from tkinter import *
import os
import pandas as pd
from movie_recomendation import trainer


main_path = os.getcwd()

movies_path = f"{main_path}/ml-1m/movies.dat"
movies_data = pd.read_csv(movies_path,names=["data"],encoding="latin-1", sep="\t")

movies = []
for i,elem in enumerate(movies_data["data"]):
    m = elem.split("::")[1]
    g = elem.split("::")[2]
    movies.append(f"{m} :: {g}")

window = Tk()
window.title('Movies recomendation - NCF | Alongoman & Yogevhad 2022')
window.geometry("500x500")
yscrollbar = Scrollbar(window)
yscrollbar.pack(side = RIGHT, fill = Y)

label = Label(window,text="Select the movies you saw and liked :  ", font=("Times New Roman", 10), padx=10, pady=10)
label.pack()
items = Listbox(window, selectmode="multiple", yscrollcommand=yscrollbar.set)


items.pack(padx=10, pady=10, expand=YES, fill="both")


for each_item in range(len(movies)):
    items.insert(END, movies[each_item])
    items.itemconfig(each_item, bg="yellow")

yscrollbar.config(command = items.yview)

def showSelected(user_movies):
    user_movies.clear()
    names = items.curselection()
    for i in names:
        text = items.get(i)
        m = text.split("::")[0]
        user_movies.append((int(i+1),m))
    print("movies selected:")
    for val in user_movies:
        print(f"\t {val[1]}")

    print("if you sure in this selction please close the window")


user_movies = []

btn = Button(window, text='Print Selected', command=lambda: showSelected(user_movies))

btn.pack(side='bottom')

window.mainloop()

print("training the model to get recomendations")

# insert best parameters here
trainer.Train(lr=1e-3, batch_size=200, embedding_size=32, epochs=20, top_k=10, num_layers=3,
              save_weights=True, show_graph=False, recomendation_num=30)