"""This files for eval private data with sentence-transformers model."""

from sentence_transformers import SentenceTransformer
import pandas as pd
import csv
import time

# Load the pre-trained model
model = SentenceTransformer("/training_nli_all-MiniLM-L6-v2-2025-07-02_09-32-22/final")


# # origin usage example
# sentences1 = "算法工程师"
# sentences2 = "算法研究员"

# embeddings_1 = model.encode(sentences1)
# embeddings_2 = model.encode(sentences2)
# similarities = 1 if model.similarity(embeddings_1, embeddings_2).item() >= 0.999 else 0
# print(similarities)


def eval_prd_data(data: pd.DataFrame, id: int):
    t1 = time.time()
    f = open(f"./{id}.csv",'a',encoding = "utf8", newline = "")
    writer = csv.DictWriter(f, fieldnames = ['sentence1','sentence2', 'score', 'predict'])
    writer.writeheader()
    for i in range(len(data)):
        label = data.loc[i, "score"]
        s1 = data.loc[i, 'sentence1']
        s2 = data.loc[i, "sentence2"]
        vec1 = model.encode(s1)
        vec2 = model.encode(s2)
        similarities = 1 if model.similarity(vec1, vec2).item() >= 0.99 else 0
        new_doc = {}
        new_doc = {
            "sentence1": s1,
            "sentence2": s2,
            "score": label,
            "predict": similarities
        }
        print(new_doc)
        writer.writerow(new_doc)
    f.close()
    t2 = time.time()
    print(f"Thread {id} finished in {t2 - t1:.2f} seconds.")
    print('done!')
    return 

if __name__ == "__main__":
    import threading
    
    # df.columns.tolist: ["sentence1", "sentence2", "score"]
    df = pd.read_csv("./eval_data/eval_73914_0703.csv")
    batch_size = 10000
    data_list = []
    for i in range(0, len(df), batch_size):
        data_list.append(df[i: i + batch_size].reset_index(drop=True))
    
    thread_list = []
    for i, data in enumerate(data_list):
        thread_list.append(threading.Thread(target=eval_prd_data, args=(data, i)))
    
    for t in thread_list:
        t.start()
    
    for t in thread_list:
        t.join()