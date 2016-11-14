### Introduction
Homograph words are words with the same spelling, but can have different meanings. The list of homograph words in English can be found in <em>homograps.md</em>. For example, as an old joke says:

> Q: Why do you think movie stars are so cool?

> A: Because they have lots of fans

The twist in here is that the words <b>"cool"</b> and <b>"fans"</b> have two different meanings. It is fairly simple for human to decide which meaning of a homograph word is used given the context where the word is in. But to train the computer to do the same is not a simple tasks. In Natural Language Processing, this challenge is call "word wense disambiguation". Several algorithms have been developed, some are simple and some are very sophisticated. In this project, I attempted to explore the word sense disambiguation challenge with different type of models.

### Data
The dataset I used for this project is obtainedfrom from the Senseval corpus where the meanings of the homographs are labeled in each example.
### Methods and Results

#### Preprocessing of Data ####
For each homograph, the example sentences and their lables are split into two lists. The documents are then lemmatized and Tf-idf vectorized using <em>scikit-learn</em> libraries. Once the text documents are converted to numeric, For this project, there are two different approaches to classify the meanings of the homographs: semi-supervised and supervised learnings.

#### 1. Semi-Supervised Learning####
This approach ignores the labled meanings of the homgraphs in their examples and attempts to classify the meanings using clusterings. The reason behind this is that if there can be words that appear mulitple times, unsupervised cluserings may be able to detect some useful patterns that can help the classification process. Then the labels were used to determine the accuracy, precision and recall rates of the models.

* <b>Kmeans</b>:
As the simplest clustering method, Kmeans can takes in the raw dataset and attempts to make a pre-defined number of clusters. In here the number of clusters are set to be the number of different meanings of the word. Unfortunately, the model does not give very meaningful predictions. The figure below shows the result for Kmeans clutering for the dataset of the word "Hard" when projected to two-dimension. We can see right away that the <em>"Curse of Dimensionality"</em> prevents our clusters to separate well.

![alt text](https://github.com/hainguyenau/Word_Sense/blob/master/kmeans%20figure.png "Kmeans Clustering")

* <b>Agglomerative Clustering</b>
Since Kmeans can only use Euclidean distances as its affinity, the next logical thing to remedy this problem is to use <em>Hierachical Clustering</em> where <em>cosine similarity</em> or <em>Jaccard similarity</em> can be used. However, this algorithm is very memory-intensive, takes very long to run and does not yield good cross-validation accuracy, precision and recall results for the hyper-parameters I used. Due to the time constraint of this project, I decided to try another model.

* <b>Linear Discriminant Analysis</b>
Again, this model is just as memory-intensive as the <em>Agglomerative</em> model. The validation results are similar to that of the <em>Agglomerative</em> model.

#### 2. Supervised Learning ####
Since all of the unsupervised methods did not yield good results, I decided to utilize the labels of the datasets to make supervised predictions.

* <b>Multinomial Naive Bayes</b>
This model is simple, yet has the potential to make good predictions. By adjusing the hyper-parameter <em>alpha</em> for the model, I found that alpha = 0.12 is optimum. The accuracy, precision and recall averages for a few different homographs are shown below


|Word | Accuracy| Precision| Recall|
|:-----:|:--------:|:----:|:----:|
|"HARD"|72%|74%|80%|
|"INTEREST"|75%|73%|75%|
|"LINE"|72%|74%|72%|


When compared the base algorithm where we always predict the most common meaning of a word (with accuracy between 51.4% and 57%), this model seems to perform better. 

### Conclusion

The <em>Multinomial Naive Bayes</em> model is simple, yet effective for the purpose of this project. Since the model can be applied as long as we have a labeled dataset, it can be extended to other languages, even non-latin languages such as Chinese or Japanese. For the time being, this model is quite bulky (we need one model for each word), but it can be improved in the future. 

### References
1. https://en.wikipedia.org/wiki/Homograph
2. https://en.wikipedia.org/wiki/Word-sense_disambiguation
3. https://www.cs.york.ac.uk/semeval2010_WSI/datasets.html
4. http://www.senseval.org/


  [1]: #file:7b2cfd55-3a68-ce62-8a2d-e61df479614c