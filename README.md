# yelp_review_rating_prediction
This project is to predict yelp review ratings only based on the review contents. The dataset is form kaggle open source, with around 2.0 million of yelp reviews. https://www.kaggle.com/yelp-dataset/yelp-dataset 

All the reviews are analyzed with NLP first. In order to build a predictive model to forcast the review ratings, there are two approaches in this project: 

1. Perform features engineering in the yelp reviews contents and apply traditional machine learning algorithms to build the models with the generated features. The techniques includes: sentiment analysis score, LDA model, text mining, part of speech(POS) and etc. 
2. Apply LSTM, one of the RNN models to train the model directly, and compare the accuracies between two approaches.  


This work can help businesses better understand their services from customers form their online reviews, and also deal with customer complaints by intelligently allocating customer service resources and better manage their online reputation.
It can also help business reduce the burden of negative reviews by flagging customers who are likely to complain online.
