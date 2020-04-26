# Twitter-text-sentiment-classification

Tried but failed to raise accuracy:
1. remove stemmed words
(this might works in other dataset, sentences on twitter are very different, sometimes they look like alien language...So I will try this methods in other dataset and see if there is different)

2. remove the comma, signals, blablabla
(I think sometimes these signals can help the lstm to learn the sentences, take them of does not help in this dataset tho)

3. center loss
(Intuitively this does not work here, i just want to give it a try:) center loss usually works in multi-classification problems. Maybe someone can try in other text sentiment classification dataset)

Tried and raised accuracy:
1. Attention
(this can focus on the hidden state of the whole sentences)

2. Bi-directional LSTM
(Bi is powerful than sigle, however in real project, the gained parameters might be a issue)

3. Combined the two different word embedding features
(w2v and fasttext from gensim, which make the feature bigger but works, could try to combine these two features with a preprocess to smaller the features?)

Afterthought:
1. boarden the word embedding feature could be a easy violent solution.
2. attention-based methods could work. Try other attentions!
3. tuning the hyperparameters would be the last things to do because it's boring and unefficient(you can try use smaller batch size)



reference:
https://colab.research.google.com/drive/16d1Xox0OW-VNuxDn1pvy2UXFIPfieCb9
