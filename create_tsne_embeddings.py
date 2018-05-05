import os
import csv
import umap
import json
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

#Not sure if we need magenta
# from magenta.models.nsynth import utils
# from magenta.models.nsynth.modelA import fastgen

np.random.seed(8)

import utils
import config

####


def get_scaled_tsne_embeddings(features,labels, perplexity, iteration):
    embedding = TSNE(n_components=2,
                     perplexity=perplexity,
                     n_iter=iteration).fit_transform(features)
    scaler = MinMaxScaler()
    scaler.fit(embedding)

    #print (scaler.transform(embedding))
    embed = scaler.transform(embedding)
    return embed


def transform_numpy_to_json(array, labels):
    data = []
    for index, position in enumerate(array):
        position_list = position.tolist()
        position_list.append(labels[index])

        data.append({
            'coordinates': position_list
        })
    return data


def get_scaled_umap_embeddings(features,labels, neighbour, distance):

    embedding = umap.UMAP(n_neighbors=neighbour,
                          min_dist=distance,
                          metric='correlation').fit_transform(features)
    scaler = MinMaxScaler()
    scaler.fit(embedding)
    embed = scaler.transform(embedding)
    return embed

def get_pca(features, labels):
    pca = PCA(n_components=2)
    transformed = pca.fit(features).transform(features)
    scaler = MinMaxScaler()
    scaler.fit(transformed)
    embed = scaler.transform(transformed)
    return embed


#####

def create_embeddings(listPath, modelA_features,modelA_labels, modelB_features, modelB_labels):

    modelA_features = np.nan_to_num(np.array(modelA_features))
    modelB_features = np.array(modelB_features)

    modelB_tuples = []
    modelA_tuples = []
    all_file_paths = []

    if(os.path.isfile(listPath)):
        with open(listPath, "r") as fp:
            for i, line in enumerate(fp):
                line = line.replace('.au\n','.wav')
                all_file_paths.append(line)

    all_json = dict()
    all_json["filenames"] = all_file_paths

    print(len(all_file_paths),
          modelA_features.shape,
          modelB_features.shape)


    tnse_embeddings_modelB = []
    tnse_embeddings_modelA = []
    perplexities = [2, 5, 30, 50, 100]
    iterations = [250, 300, 350, 400, 500]
    # perplexities = [2, 5]
    # iterations = [250, 300]
    for i, perplexity in enumerate(perplexities):
        for j, iteration in enumerate(iterations):
            print ("Perplexity : ",perplexity," Iterations :", iteration)
            tsne_modelB = get_scaled_tsne_embeddings(modelB_features,
                                                    modelB_labels,
                                                    perplexity,
                                                    iteration)
            tnse_modelA = get_scaled_tsne_embeddings(modelA_features,
                                                      modelA_labels,
                                                      perplexity,
                                                      iteration)
            tnse_embeddings_modelB.append(tsne_modelB)
            tnse_embeddings_modelA.append(tnse_modelA)

            modelB_key = 'tsnemodelB{}{}'.format(i, j)
            modelA_key = 'tsnemodelA{}{}'.format(i, j)

            all_json[modelB_key] = transform_numpy_to_json(tsne_modelB,modelB_labels)
            all_json[modelA_key] = transform_numpy_to_json(tnse_modelA,modelA_labels)

    # fig, ax = plt.subplots(nrows=len(perplexities),
    #                        ncols=len(iterations),
    #                        figsize=(30, 30))

    # for i, row in enumerate(ax):
    #     for j, col in enumerate(row):
    #         current_plot = i * len(iterations) + j
    #         col.scatter(tnse_embeddings_modelB[current_plot].T[0],
    #                     tnse_embeddings_modelB[current_plot].T[1],
    #                     s=1)
    # plt.show()


    umap_embeddings_modelB = []
    umap_embeddings_modelA = []
    neighbours = [5, 10, 15, 30, 50]
    distances = [0.000, 0.001, 0.01, 0.1, 0.5]
    # neighbours = [5]
    # distances = [0.000, 0.001]
    for i, neighbour in enumerate(neighbours):
        for j, distance in enumerate(distances):
            print ("neighbour : ",neighbour," distance :", distance)
            umap_modelB = get_scaled_umap_embeddings(modelB_features,
                                                    modelB_labels,
                                                    neighbour,
                                                    distance)
            umap_modelA = get_scaled_umap_embeddings(modelA_features,
                                                      modelA_labels,
                                                      neighbour,
                                                      distance)
            umap_embeddings_modelB.append(umap_modelB)
            umap_embeddings_modelA.append(umap_modelA)

            modelB_key = 'umapmodelB{}{}'.format(i, j)
            modelA_key = 'umapmodelA{}{}'.format(i, j)

            all_json[modelB_key] = transform_numpy_to_json(umap_modelB,modelB_labels)
            all_json[modelA_key] = transform_numpy_to_json(umap_modelA,modelA_labels)



    pca_modelB = get_pca(modelB_features,modelB_labels)
    pca_modelA = get_pca(modelA_features,modelA_labels)

    modelB_key = 'pcamodelB'
    modelA_key = 'pcamodelA'

    all_json[modelB_key] = transform_numpy_to_json(pca_modelB,modelB_labels)
    all_json[modelA_key] = transform_numpy_to_json(pca_modelA,modelA_labels)


    json_name = "data.json"
    json_string = "d = '" + json.dumps(all_json) + "'"
    with open(json_name, 'w') as json_file:
        json_file.write(json_string)




predicted_prob,y_data,num_frames_test = utils.load_h5(config.SOFTMAX_RESULT_FILE)

create_embeddings(config.ALL_SONGS_PATHS,predicted_prob,y_data,predicted_prob,y_data)
