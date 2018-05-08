# Music exploration using machine learning

Music is complex and varies widely in beats, frequencies and structure. This experiment uses the spectrogram of common songs to organize thousands of everyday tracks. We implement various neural network models to train classifiers to segregate neighboring songs with similar features from the mel-spectrogram of the songs. The softmax output of the neural network thus trained is used as a reduced feature set in order to perform dimensionality reduction of the songs. We then create different 2D embeddings of the reduced feature set using various dimensionality reduction techniques. These embeddings are used to place similar song features closer together on a visually interactive song-map. This, in our research so far, is the first attempt at producing a visual and interactive map of music at a full song collection scale that users can use in order to explore the world of music, discover and listen to songs of their preference.

## 1. Methodology

In order to achieve the goal of scalability classifying thousands of songs while not loosing its transitive nature, our ap-proach does not perform a one-hot classification on the data. We also develop a means to reduce the size of the musicalfeatures from the raw audio of the song using mel-spectogram. We then train a neural network classifier on the reduced dataand use the softmax output in order to perform the visualization. The methodology we used is detailed below.

### 1.1. Audio Feature Extraction using Mel-Spectogram

This process converts the song file into a 1366 x 966 dimension vector. In the pre-processing step, Mel-Spectrograms aregenerated in a similar manner to [11] whereby we use 96 mel-bins with 256 frame hops at sampling rate of 12000Hz, withthe Fast Fourier Transform window size of 512 frames around each frame. For the purposes of this project, we use 30 secondclips of songs in order to reduce computation time. Since the exact multiple of audio frames given the above hops,windowsand sampling rates for 30 seconds, comes out to be exactly 29.12 seconds, we trim 0.44 seconds of audio from the first andthe last portion of the clip. We use the audio processing tool developed by McFee et al [25] in order to perform the requiredmanipulation on the song data. The output of the pre-processing step is a song vector of dimension 1366 x 96 x (total numberof songs) which we feed into the neural network model in the subsequent section.



![Alt text](https://github.com/codeJRV/MusicMapz/blob/master/README-FILES/melspec.png?raw=true )

```Figure 1: Mel-Spectogram of a song```

### 1.2. Song Feature Extraction and Classification

The first step towards generating the map of songs is to extract meaningful features out of them. These would be featureslike song genre, beats, timbre, language, artists etc. The raw data of the song and its meta-data would be fed into a neuralnetwork and trained using classification labels available. In order to do this we used convolutional recurrent neural networl toclassify the songs into genres. We then looked for interesting patterns in the outputs of the softmax layer. This approach hasalready been tried out in neuroscience and vision [29] and we believe using the flattened features of the softmax/penultimatelayers of neural networks in music will also provide us with a reduced feature set that can be fed into our dimensionalityreduction later on. In order to test this approach, we generate a t-SNE plot directly of the Mel-spectogram of songs. Figure[3] shows that the raw Mel-spectogram does not provide us with a meaningful clustering of songs.


![Alt text](https://github.com/codeJRV/MusicMapz/blob/master/README-FILES/t_sne_melspectrogram.png?raw=true )


```Figure 2: Performing T-SNE of the mel-spectograms of the songs directly does not lead to distinct or distinguishable results```


![Alt text](https://github.com/codeJRV/MusicMapz/blob/master/README-FILES/tnse_CRNN_features.png?raw=true )

```Figure 3: Performing T-SNE of the mel-spectograms of the songs using the softmax output results in better classification and clustering```


![Alt text](https://github.com/codeJRV/MusicMapz/blob/master/README-FILES/model1.png?raw=true )



```Figure 4: Final CRNN model that gives 80% accuracy on test data```

Thus, in order to achieve meaningful dimensionality reduction, the song features are decomposed into a 10 dimensionalvector using a convolutional recurrent neural network. We experimented with the size and depth of the neural network inorder to develop a suitable classifier for genre prediction, with a targeted accuracy of atleast 0.5. We tried models from [11]and [?] and finally built our own model. Using the right features are very important to identify genres [38] or in our casethe right feature vectors of the songs. Here we need to balance out the specificity, ie. response to particular type of songand the generality, ie. the amount of spread of feature vectors when the songs are fed into the system, so that we get adecent but relevant spread in our final song-map. If the outputs of the song classifier are too specific, the final dimensionalityreduction will have a lot of data points clustered close together and there will not be sufficient scatter in the data-points inthe visualization. On the other hand data that is poorly classified will have a lot of intermixed points which will lead to poor relevance of one song to other. Thus the aim of the project is not to exceed the performance metric of the existingclassification systems, but to tune the results of the neural networks to generate an appropriate set of feature vectors, whichupon being fed into the dimensionality reduction system provides us with a map of music which will be relevant to the user.

The final model we developed was a 8 layer Gated Recurrent Convolutional Neural Network with 6 convolutional layersand 2 recurrent layers. The convolutional layers extract high level features from the neural network while the recurrent layersprovide the ability to capture time-series information from the data. In this manner, features such as beats, treble, and vocalscan be thought of as a series of neuron activations in a sequential manner, with the convolutional layers aggregating lowerlevel features into higher level features, and the recurrent units firing, if a particular high-level feature is sustained for a fixedperiod of time, for example legato, staccato and vibrato in vocals[1]. Paper [6] explains some of these patterns in violinmusic.Even after mel-spectogram reduction, the size of the input was very large. Thus we optimized the algorithms in orderto reduce the computation time. We converted the tensorflow backend of keras to GPU based code in order to reducecomputation time from approximately 5 minutes for 1 epoch of 1000 songs, to around 40 seconds per epoch for 1000 songs.This was done by lowering the batch size and running the computation using an Nvidia GTX1080x GPU. This allowed us toscale up the data collection from 400 to 1000 to upto 23,000 songs.A table detailing the architecture of various models and their classification accuracy, with respect to various hyper-parameters are given.

```Model          Dataset    Epochs Train_Acc Validation_Acc Test_Acc.```
```4 conv. layers GTZAN       40      0.87       0.8            0.78```
```4 conv. layers GTZAN       300     0.89       0.78           0.75```
```6 conv. layers GTZAN       100     0.93       0.65           0.6 ```
```6 conv. layers GTZAN+FMA   100     0.90       0.80           0.80```

```Table 1: Accuracies for different neural network models```

### 1.3. Visualization

There are multiple methods to perform dimensionality reduction. Principal Component Analysis [36] and Linear Discrim-inant Analysis [19] are two commonly used methods to reduce multidimensional features into two or three dimensions.PCAprovides an intuitive understanding of the music map for a starting point.In our experiment(Fig. 6(a)),the transitions of var-ious genres of music form two main outward branches, one being Metal and the other being Hip-hop. Pop music graduallytransitions into hip-hop, while rock music transitions into metal. Classical, country and reggae music appear as three branchesforming the base of the Y stem which merges in the center. This trend is still valid on music on 1000 song dataset or the23,000 song dataset.

However, PCA as shown in Fig 6.b also indicates that the map becomes less intuitive as the dataset size increases and thenumber of available songs in one or two genres have a larger majority than the other. Thus PCA, though providing an initialunderstanding of the way music is comprised, does not scale well to large datasets due to its many inherent limitations.Forexample, PCA is a bad choice because it is a linear algorithm which cannot distinguish non-linear structures in the data. Anin-depth study of these limitations is performed by [28], albeit for a different use case. Hence we are exploring the use oft-SNE [24] and UMAP [26] in order to develop the music map.These techniques perform much better than PCA for largesongs, as can be seen in the results section(Figure 10).

```Figure 6:```

![Alt text](https://github.com/codeJRV/MusicMapz/blob/master/README-FILES/PCA1000songs.png?raw=true )

```(a) PCA on GTZAN dataset of1000 songs```

![Alt text](https://github.com/codeJRV/MusicMapz/blob/master/README-FILES/PCA25000songs.png?raw=true )

```(b) PCA on GTZAN+FMAdataset of 23,141 songs```

 PCA plots of 10 different genres of music show that this method is not scalable, although it is easy to comprehendthe transition between different genres of music.


 ### 1.4. User Interface


 Finally we use the music map generated by the visualization into an interactive user interface that allows the user to clickand play the songs in the area selected. To accomplish this task, we used the popular web browser rendering tool Three.js[14] in order to render the songs. The input to the web application is a json file of coordinates and paths of each song in anested list dictionary structure. These ’embeddings’ are generated for all the songs present in the database, in order to getthe best possible visual output. The points encode a GET request to the URL of the song in the database, which is playedback using the Web Audio API [10]. We also used another extension to Web Audio API called webaudiox.js in order to addadditional functionality such as audio volume normalization, crossfading and sound smoothing in order to make the songsmore appealing to the listener. Note that all songs of a particular genre have the same colour. The web page was designed using a template from [4].

![Alt text](https://github.com/codeJRV/MusicMapz/blob/master/README-FILES/UI.png?raw=true )


 ``` Figure 7: User interface of our application ```


 ## 2. Datasets Used


 There are millions of songs in the world to chose from and add to our system. With an impartial feature classifier, we hopeto get better and better at mapping songs to the point where any new song released can be added to the system and it will putthe song into the right spot on our map. However, given the limitations of time and computing power, we start with smallersong sets and learn the features on smaller subsets of music. Incrementally, we worked on the following datasets :

 1. GTZAN Dataset : This dataset is the most-used public dataset for evaluation in machine listening research for music    genre recognition (MGR). It was first used for the well known paper in genre classification [33]. The dataset consists    of 1000 audio tracks each 30 seconds long distributed among 10 genres, each represented by 100 tracks. The tracks    are all 22050Hz Mono 16-bit audio files in .wav format.

 ```dataset clips   genres length[s] size[GiB]```
  
 ```small  8,000   8          30       7.6```
 ```medium 25,000  16         30       23 ```
 ```large  106,574 161        30       98 ```
 ```full   106,574 161        278      917```


 Table 2: FMA data subsets


![Alt text](https://github.com/codeJRV/MusicMapz/blob/master/README-FILES/MusicDistribution.png?raw=true )


 ```Figure 8: Distribution of no.of tracks per Genre in FMA dataset```


![Alt text](https://github.com/codeJRV/MusicMapz/blob/master/README-FILES/data_distribution.png?raw=true )

 ```Figure 9: Distribution of no.of tracks per Genre used in our music map after filtering```


 2. Free Music Archive Dataset: We wanted to train our Neural network model on relatively bigger dataset than GTZAN.    Although Million Song Dataset (MSD)[7] as well as the newer AudioSet and AcousticBrainz are very large-scale    reference datasets, they do not provide raw mp3 files and downloading the mp3 files in itself is a challenge due to    copyright issues. FMA provides an excellent alternative as a medium scale music dataset of more than 100k songs    with freely available full-length and high-quality audio. It provides pre-computed features, together with track- and    user-level meta-data, and tags. The dataset is divided and maintained in subsets listed in Table 2.The problem with the FMA dataset was that there was a non-uniform distribution of songs in different genres(Figure 8).Hence, we merged both GTZAN and FMA datasets together and selected 10 popular genres and filtered the songs to be usedfor our visualization. Distribution of trakcs per genre was still dominated by few genres but it has helped us further ouranalysis on Genres selected from GTZAN dataset (Figure 9). Finally, the filtered dataset consisted of 23,141 songs whichwere divided among training testing and validation set, with 14,809 songs for training, 3,703 for validation and 4,629 fortesting.

 ## 3. Results

 Plotting songs on a 2D map gave us many auditory insights which are hard to describe on paper and vary from person toperson. On the other hand, visual properties are easier to describe in general. For example, the t-SNE algorithm has a hyper-parameter - Perplexity, which balances attention between local and global structures of the data. It specifies the number ofclose neighbors each point has. We observed that as we increase the perplexity value we started getting more visually distinctclusters of music genres.UMAP has two hyper-parameters: no-of-neighbors, which indicates the number of neighboring points in the local ap-proximations of the manifold structure. If more neighbors are considered, lower dimension preserves global structure in dataset. 

 The second parameter is min-dist, which controls how tightly the embedding is allowed to compress points together.If compressing is allowed with higher margin it ensures that the embedded points are more evenly distributed and retain theglobal structure in dataset.It was observed that with larger values of both no-of-neighbors and min-dist we started getting much better clusters ofmusic genres on our dataset. In conclusion, we found that the distribution of points by UMAP is more visually appealing thant-SNE to most viewers. Figure 10 shows the various outputs generated by the system for varying parameter combinations.


![Alt text](https://github.com/codeJRV/MusicMapz/blob/master/README-FILES/Results.png?raw=true )

```Figure 10: Music Maps generated using different t-SNE and UMAP parameters```

 ## 4. Conclusion

 We present a novel method of visualizing music using the latest state-of-art visualization techniques, powered by a custombuilt neural network that matches the current state-of-art in classification accuracy, and is optimized to run an epoch of 1000songs in under a minute. This enables us to develop the first mapping of music on a full-song collection level in currentliterature, to the best of our knowledge. The above methodology is successfully demonstrated on a dataset of 23,000 audiotracks which demonstrates the viability of the idea that the softmax output of the neural network can be used as a reducedfeature set for song visualization. The outputs of the visualization algorithms, are not only mysterious but also beautiful,presenting the audience with an unprecedented way of understanding music, opening up a new dimension in how music is perceived and on a larger scale, interacted with. This, we believe is the greatest outcome of the project.

 ## References
 [1] 3 voice techniques: Legato, staccato, vibrato: https://www.youtube.com/watch?v=shtz01-zie8.
 
 [2] Data to date: the rapid rise of social and streaming: https://www.nextbigsound.com/industry-report/2015.
 
 [3] Google infinite drum machine: https://experiments.withgoogle.com/ai/drum-machine.
 
 [4] Umap tsne embedding visualiser: https://github.com/fedden/umap-tsne-embedding-visualiser.
 
 [5] G. Adomavicius and A. Tuzhilin. Toward the next generation of recommender systems: A survey of the state-of-the-art and possibleextensions.IEEE transactions on knowledge and data engineering, 17(6):734–749, 2005.
 
 [6] L. Aversano. Terminologia violinistica tra sei e settecento.Tra le note, pages 1000–1034, 1996.
 
 [7] T. Bertin-Mahieux, D. P. Ellis, B. Whitman, and P. Lamere. The million song dataset. InIsmir, volume 2, page 10, 2011.
 
 [8] K. Bunte, M. Biehl, and B. Hammer. A general framework for dimensionality-reducing data visualization mapping.Neural Compu-tation, 24(3):771–804, 2012.
 
 [9]O. Celma, M. Ramirez, and P. Herrera. Foafing the music: A music recommendation system based on rss feeds and user preferences.Inin ISMIR. Citeseer, 2005.
 
 [10] H. Choi and J. Berger. Waax: Web audio api extension. InNIME, pages 499–502, 2013.
 
 [11] K. Choi, G. Fazekas, M. Sandler, and K. Cho. Convolutional recurrent neural networks for 
 music classification. InAcoustics, Speechand Signal Processing (ICASSP), 2017 IEEE International Conference on, pages 2392–2396. IEEE, 2017.
 
 [12] M. Defferrard, K. Benzi, P. Vandergheynst, and X. Bresson. Fma: A dataset for music analysis.arXiv preprint arXiv:1612.01840,2016.
 
 [13] S. Dieleman and B. Schrauwen. Multiscale approaches to music audio feature learning. In14th International Society for MusicInformation Retrieval Conference (ISMIR-2013), pages 116–121. Pontifıcia Universidade Catolica do Parana, 2013. 
 
 [14] J. Dirksen.Learning Three. js: the JavaScript 3D library for WebGL. Packt Publishing Ltd, 2013.
 
 [15] P. Filzmoser, K. Hron, and C. Reimann. Univariate statistical analysis of environmental (compositional) data: problems and possibil-ities.Science of the Total Environment, 407(23):6100–6108, 2009.
 
 [16] S. Flesch, A.-S. Gutsche, and D. Paschen. Exploring the untapped potential of sound maps.
 
 [17] J. Foote. Visualizing music and audio using self-similarity. InProceedings of the seventh ACM international conference on Multime-dia (Part 1), pages 77–80. ACM, 1999.
 
 [18] W. Hill, L. Stead, M. Rosenstein, and G. Furnas. Recommending and evaluating choices in a virtual community of use. InProceedingsof the SIGCHI conference on Human factors in computing systems, pages 194–201. ACM Press/Addison-Wesley Publishing Co.,1995.
 
 [19] A. J. Izenman. Linear discriminant analysis. InModern multivariate statistical techniques, pages 237–280. Springer, 2013.
 
 [20] F.-F. Kuo, M.-F. Chiang, M.-K. Shan, and S.-Y. Lee. Emotion-based music recommendation by association discovery from filmmusic. InProceedings of the 13th annual ACM international conference on Multimedia, pages 507–510. ACM, 2005.
 
 [21] S. W. Lee, J. Bang, and G. Essl. Live coding youtube: Organizing streaming media for an audiovisual performance.Ann Arbor,1001:48109–2121, 2017.
 
 [22] M. Levy and M. Sandler. A semantic space for music derived from social tags.Austrian Compuer Society, 1:12, 2007.
 
 [23] T. Li, M. Ogihara, and Q. Li. A comparative study on content-based music genre classification. InProceedings of the 26th annualinternational ACM SIGIR conference on Research and development in informaion retrieval, pages 282–289. ACM, 2003.
 
 [24] L. v. d. Maaten and G. Hinton. Visualizing data using t-sne.Journal of machine learning research, 9(Nov):2579–2605, 2008.
 
 [25] B. McFee, C. Raffel, D. Liang, D. P. Ellis, M. McVicar, E. Battenberg, and O. Nieto. librosa: Audio and music signal analysis inpython. InProceedings of the 14th python in science conference, pages 18–25, 2015.
 
 [26] L. McInnes and J. Healy. Umap: Uniform manifold approximation and projection for dimension reduction. arXiv preprintarXiv:1802.03426, 2018.
 
 [27] H.-S. Park, J.-O. Yoo, and S.-B. Cho. A context-aware music recommendation system using fuzzy bayesian networks with utilitytheory. InInternational conference on Fuzzy systems and knowledge discovery, pages 970–979. Springer, 2006.
 
 [28] S. Prasad and L. M. Bruce. Limitations of principal components analysis for hyperspectral target recognition.IEEE Geoscience andRemote Sensing Letters, 5(4):625–629, 2008.
 
 [29] S. Rohit and S. Chakravarthy. A convolutional neural network model of the neural responses of inferotemporal cortex to complexvisual objects.BMC neuroscience, 12(1):P35, 2011.
 
 [30] S. Saxena and C. J. Romanowski. Theme extraction from lyrics.
 
 [31] Y. Song, S. Dixon, and M. Pearce. A survey of music recommendation systems and future perspectives. In9th InternationalSymposium on Computer Music Modeling and Retrieval, volume 4, 2012.
 
 [32] M. Torrens, P. Hertzog, and J. L. Arcos. Visualizing and exploring personal music libraries. InISMIR, 2004.
 
 [33] G. Tzanetakis and P. Cook. Musical genre classification of audio signals. IEEE Transactions on speech and audio processing,10(5):293–302, 2002.
 
 [34] A. Van den Oord, S. Dieleman, and B. Schrauwen. Deep content-based music recommendation. InAdvances in neural information processing systems, pages 2643–2651, 2013.
 
 [35] M. M. Van Hulle. Self-organizing maps. InHandbook of Natural Computing, pages 585–622. Springer, 2012.
 
 [36] S. Wold, K. Esbensen, and P. Geladi. Principal component analysis.Chemometrics and intelligent laboratory systems, 2(1-3):37–52,1987.
 
 [37] Y. Xia, L. Wang, and K.-F. Wong. Sentiment vector space model for lyric-based song sentiment classification.International Journalof Computer Processing Of Languages, 21(04):309–330, 2008.
 
 [38] E. Zheng, M. Moh, and T.-S. Moh. Music genre classification: A n-gram based musicological approach. InAdvance ComputingConference (IACC), 2017 IEEE 7th International, pages 671–677. IEEE, 2017.