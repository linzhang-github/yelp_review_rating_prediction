


import pandas as pd
import numpy as np
import string
import os
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


## segement reviews based on their categories 


df = pd.read_csv("CleanedYelpData.csv",dtype=object)


# In[90]:


df = df.dropna(subset = ['categories']) 


# In[92]:


ref = pd.read_excel("Category_List.xlsx")
ref.head()


# In[91]:


df = df.head(1000000) 


# In[93]:


ref = ref.groupby(ref.Categories)['SubCategories'].agg(lambda col: ', '.join(col))
ref = pd.DataFrame(ref)
ref=ref.reset_index()



Reference = {}

for i in range(ref.shape[0]):
    Reference[ref.iloc[i,0]] = ref.iloc[i,1].split(", ")


df['cat_list'] = [i.split(";") for i in df.categories]





def cat_frequency(cell):
    ## create a dic with all the keys in Reference but values are all 0 
    newDict = {key: 0 for key in Reference.keys()}
    for i in cell:
        for j in Reference:
            if i in Reference[j]:
                ## then if the cat in df shows up in the reference values, the value of new dic plus 1 
                newDict[j] = newDict[j] + 1 
            else:
                continue
    return newDict


df['cat_freq'] = list(map(cat_frequency, df.cat_list))




def find_freq_cat(cell):
        return max(cell, key=lambda k: cell[k])

df['cat_max'] = list(map(find_freq_cat,df.cat_freq))




df = df[['review_id', 'user_id', 'business_id', 'review_rating', 'date', 'text',
       'useful', 'funny', 'cool', 'name', 'neighborhood', 'address', 'city',
       'state', 'postal_code', 'latitude', 'longitude', 'stars',
       'review_count', 'is_open', 'cat_max', 'text_nopunct', 'text_lower',
       'text_nostopwords']] 





# df.head(5)
df.rename(columns = {'cat_max':'category'}, inplace = True)


# In[141]:


df_cate = df.category.value_counts().to_frame().reset_index()
df_cate.rename(columns = {'index':'category', 'category':'number_of_review'}, inplace = True)





df_restaunt = df[df.category == 'Restaurants']



## Clean the text 


all_review=df_restaunt_final['text'].tolist
all_review_clean = [i.replace('\n','').replace('\r', '').replace("'","").replace('"','') for i in all_review] 

### Convert to lower case
all_review_clean2 = [i.lower() for i in all_review_clean]



# Remove punctua
import string
all_review_clean3 = [i.translate(str.maketrans('','', string.punctuation)) for i in all_review_clean2]
# all_review_clean3


# In[24]:


# Tokenize 
all_review_tokenize = [nltk.word_tokenize(i) for i in all_review_clean3]
all_review_tokenize


# In[25]:


# Limit the length of words between 3 and 15
all_review_tokenize2 = []
for i in all_review_tokenize:
    all_review_tokenize2.append([w for w in i if 15>len(w)>3])


# In[26]:


# Remove stop words and stem 
ps = PorterStemmer()
stop = set(stopwords.words('english'))
# starr_words = ['starr','companies','starrcompanies','email','thank','thanks','mail', 'please', 'attach','attachment','attached']
doc_stemmed = []
for one_doc in all_review_tokenize2:
    doc_filtered = [w for w in one_doc if not w in stop]
#     doc_filtered2 = [w for w in doc_filtered if not w in starr_words]
    doc_2 = [w for w in doc_filtered if w.isalpha()]#remove ",", "."
    doc_stem = [ps.stem(w) for w in doc_2]
    doc_stemmed.append(doc_stem)




from gensim import corpora, models
import gensim
dictionary = corpora.Dictionary(doc_stemmed) 


corpus = [dictionary.doc2bow(text) for text in doc_stemmed] 

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, id2word = dictionary, passes = 20,alpha='auto',minimum_probability=0.0 )
ldamodel.show_topics()





import gensim
from gensim import corpora, models
import pyLDAvis.gensim 

pic_res =  pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
pyLDAvis.display(pic_res) 




topic_weight = []
for i, row in enumerate(ldamodel[corpus]):
    row = sorted(row, key=lambda x: (x[1]), reverse=True)
    topic_weight.append(row)
    print (i,row)
    
#     for j, (topic_num, prop_topic) in enumerate(row):
#         wp = ldamodel.show_topic(topic_num)
#         print (wp[:10])
#     print ('\n')


# In[205]:


sorted_topic_weight = [sorted(i, key=lambda j: j[0]) for i in topic_weight]


# In[206]:


# create a separate dataframe of topic weights
topic_weight_df = pd.DataFrame()


# In[207]:


topic_weight_0 = []
topic_weight_1 = []
topic_weight_2 = []
topic_weight_3 = []
topic_weight_4 = []
# topic_weight_5 = []
for tuple_list in sorted_topic_weight:
    for tuple_element in tuple_list:
        if tuple_element[0] == 0:
            topic_weight_0.append(tuple_element[1])
        if tuple_element[0] == 1:
            topic_weight_1.append(tuple_element[1])
        if tuple_element[0] == 2:
            topic_weight_2.append(tuple_element[1])
        if tuple_element[0] == 3:
            topic_weight_3.append(tuple_element[1])
        if tuple_element[0] == 4:
            topic_weight_4.append(tuple_element[1])
        
topic_weight_df['Topic_1'] = topic_weight_0
topic_weight_df['Topic_2'] = topic_weight_1
topic_weight_df['Topic_3'] = topic_weight_2
topic_weight_df['Topic_4'] = topic_weight_3
topic_weight_df['Topic_5'] = topic_weight_4
# topic_weight_df['Topic_6'] = topic_weight_5


topic_weight_df['mergeID'] = range(0,len(topic_weight_df))


# In[213]:




# # df_downsampled_and_topicweight.head()


df_restaunt['mergeID'] = range(0,len(df_restaunt))
df_restaunt_topics = pd.merge(df_restaunt, topic_weight_df, on='mergeID')




df_restaunt_final = df_restaunt_with_topic[['review_rating','Topic_1', 'Topic_2', 'Topic_3','Topic_4', 'Topic_5']]




from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
from sklearn import metrics
from sklearn.model_selection import train_test_split
# Separate input features (X) and target variable (y)
y = df_restaunt_final.review_rating
X = df_restaunt_final.drop('review_rating', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



NB_model = MultinomialNB().fit(X_train, y_train)
NB_pred_y_0 = NB_model.predict(X_test) 



y_pred = NB_model.predict(X_test) 

NB_model.score(X_test, y_test) #0.37099615631005767 





from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
logreg = LogisticRegression(multi_class='multinomial',solver ='newton-cg')
logreg.fit(X_train, y_train)


clf_0 = LogisticRegression().fit(X, y)
 
# Predict on training set
pred_y_0 = clf_0.predict(X) 

# How's the accuracy?
print( accuracy_score(pred_y_0, y) ) 

y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test))) ## 44%


# In[85]:


logreg.coef_




from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)




a = [62,76,157,275,219]
b = [10185,9354,14448,28339,36789]


print("Rating 1 accuracy prediciton is  " + str(100 *(62+sum(b)-10185)/(sum(a)+sum(b))) + "%")
print("Rating 2 accuracy prediciton is  " + str(100 *(76+sum(b)-9354)/(sum(a)+sum(b))) + "%") 
print("Rating 3 accuracy prediciton is  " + str(100 *(157+sum(b)-14448)/(sum(a)+sum(b))) + "%")
print("Rating 4 accuracy prediciton is  " + str(100 *(275+sum(b)-28339)/(sum(a)+sum(b))) + "%") 
print("Rating 5 accuracy prediciton is  " + str(100 *(219+sum(b)-36789)/(sum(a)+sum(b))) + "%") 




from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) # 0.42240550928891735 


# ## Basline of different ratings

# In[57]:


print("Rating 1 is  " + str (100*10247/len(y_test)) + ' %' + " of total")
print("Rating 2 is  " + str (100*9430/len(y_test)) + ' %' + " of total") 
print("Rating 3 is  " + str (100*14605/len(y_test)) + ' %' + " of total")  
print("Rating 4 is  " + str (100*28614/len(y_test)) + ' %' + " of total")   
print("Rating 5 is  " + str (100*37008/len(y_test)) + ' %' + " of total")   


# In[ ]: