#!/usr/bin/env python
# coding: utf-8

# # AI BASED SOLUTION FOR FLAGGING OF FALSE INFORMATION ON ONLINE PLATFORMS 

# In[1]:


import pandas as pd
import numpy as np
data = pd.read_csv("E:/file2/Desktop/new_newsdesk.csv")


# In[2]:


data = data.dropna(how = 'any', axis = 0)


# In[3]:


data.isnull().sum()


# In[4]:


data.label.value_counts()


# In[5]:


from nltk.stem.porter import PorterStemmer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk.stem import WordNetLemmatizer


# In[9]:


from nltk.corpus import stopwords
import nltk


# In[10]:


stemming = PorterStemmer()
lemmatizer = WordNetLemmatizer()


# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


X=data[['text']]
Y=data['label']


# In[116]:


X


# In[115]:


p=data['text']
print(p)


# In[13]:


x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)


# In[14]:


print('x_train:',x_train.shape)
print('y_train:',y_train.shape)
print('x_test:',x_test.shape)
print('y_test:',y_test.shape)


# In[15]:


X_train = x_train


# In[16]:


x_train.head()


# In[17]:


X_test = x_test


# In[18]:


y_train.head()


# In[19]:


X_train.head()


# In[20]:


X_test.head(10)


# In[ ]:





# In[21]:


y_test


# # Data Preprocessing

# In[22]:


def preprocess(pro):
    process = re.sub('[^a-zA-Z]'," ",pro)
    lowe = process.lower()
    tokens = lowe.split()
   
    stop = [lemmatizer.lemmatize(i) for i in tokens if i not in stopwords.words('English')]
    lemmas =pd.Series([ " ".join(stop),len(stop)])
    return lemmas


# In[23]:


px_train = X_train['text'].apply(preprocess)


# In[109]:


px_train.head()


# In[110]:


type(px_train)


# # Test data preprocessing

# In[26]:


px_test = X_test['text'].apply(preprocess)


# In[27]:


px_test.head()


# In[28]:


px_test.columns = ['clean_text','text_length']
px_test.head()


# In[29]:


px_train.columns = ['clean_text','text_length']
px_train.head()


# In[30]:


X_train = pd.concat([X_train,px_train],axis=1)
X_train.head()


# In[31]:


X_test = pd.concat([X_test,px_test],axis=1)


# In[32]:


X_test.head()


# In[33]:


from wordcloud import WordCloud


# In[34]:


y_train


# In[35]:


y_test


# In[36]:


real_n = X_train.loc[y_train=='REAL', :]
real_n.head()


# In[37]:


words = ' '.join(real_n['clean_text'])
clean_word = " ".join([word for word in words.split()])


# In[38]:


real_word = WordCloud(stopwords=stopwords.words("english"),
                     background_color='black',
                     width=1600,
                     height=800).generate(clean_word)


# In[39]:


plt.figure(1,figsize=(30,20))
plt.imshow(real_word)
plt.axis('off')
plt.show()


# In[40]:


fake_n = X_train.loc[y_train=='FAKE', :]
fake_n.head()


# In[41]:


words_f = ' '.join(fake_n['clean_text'])
clean_word_f = " ".join([word for word in words_f.split()])


# In[42]:


real_word_f = WordCloud(stopwords=stopwords.words("english"),
                     background_color='black',
                     width=1600,
                     height=800).generate(clean_word_f)


# In[43]:


plt.figure(1,figsize=(30,20))
plt.imshow(real_word_f)
plt.axis('off')
plt.show()


# # Tfidf Vectorizer

# In[44]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[45]:


tf_vector = TfidfVectorizer()


# In[46]:


X_train_t = tf_vector.fit_transform(X_train['clean_text'])


# In[47]:


(X_train_t)


# In[48]:


print('unique words:',len(tf_vector.vocabulary_))
print('Shape of input data:',X_train_t.shape)


# # Test data
# 

# In[49]:


X_test_tf = tf_vector.transform(X_test['clean_text'])


# In[50]:


X_test_tf


# # Label Encoding

# In[51]:


label = LabelEncoder()


# In[52]:


y_train = label.fit_transform(y_train)


# In[53]:


y_train


# In[54]:


Y_test = label.transform(y_test)


# In[55]:


Y_test


# # Logistic Regression Model

# In[56]:


from sklearn.linear_model import LogisticRegression


# In[57]:


models = LogisticRegression()


# In[58]:


models.fit(X_train_t,y_train)


# In[59]:


from sklearn.metrics import accuracy_score


# In[103]:


l_train_score = models.predict(X_train_t)
l_train_accuracy = accuracy_score(l_train_score,y_train)


# In[111]:


print('train_accuracy:',l_train_accuracy)


# In[105]:


l_test_score = models.predict(X_test_tf)


# In[106]:


l_test_accuracy = accuracy_score(test_score,Y_test) 


# In[107]:


print('test_acccuracy:',l_test_accuracy)


# In[108]:


cmx_1=confusion_matrix(Y_test,l_test_score)
print("\nNo. of test samples : ",len(X_test))
print("\n Confustion Matrix : \n",cmx_2)
print("\nPerfomance measures are: \n",classification_report(Y_test, l_test_score))


# In[ ]:





# In[65]:


news=X_train_t[1]


# In[66]:


prediction = models.predict(news)
print(prediction)

if (prediction[0]==0):
    print('The news is fake')
else:
    print('The news is real')


# In[67]:


from sklearn import metrics


# In[68]:


confusion = metrics.confusion_matrix(Y_test, test_score)


# In[69]:


confusion


# # SVM

# In[70]:


from sklearn.svm import SVC


# In[71]:


support = svm.SVC()


# In[72]:


support


# In[73]:


support.fit(X_train_t,y_train)


# In[74]:


train_score_1 = support.predict(X_train_t)
train_accuracy_1 = accuracy_score(train_score_1,y_train)


# In[75]:


print('train_accuracy:',train_accuracy_1)


# In[76]:


test_score_1 = support.predict(X_test_tf)


# In[77]:


test_accuracy_1 = accuracy_score(test_score_1,Y_test)


# In[78]:


print('test_acccuracy:',test_accuracy_1)


# In[79]:


news_1=X_train_t[1]


# In[80]:


prediction_1 = support.predict(news_1)
print(prediction_1)

if (prediction_1[0]==0):
    print('The news is fake')
else:
    print('The news is real')


# In[81]:


from sklearn.metrics import classification_report, confusion_matrix


# In[82]:


confusion = metrics.confusion_matrix(Y_test, test_score_1)


# In[83]:


cmx=confusion_matrix(Y_test,test_score)
print("\nNo. of test samples : ",len(X_test))
print("\n Confustion Matrix : \n",cmx)
print("\nPerfomance measures are: \n",classification_report(Y_test, test_score))


# # KNN

# In[85]:


from sklearn.neighbors import KNeighborsClassifier


# In[94]:


knn_model = KNeighborsClassifier(n_neighbors=5)


# In[95]:


knn_model.fit(X_train_t,y_train)


# In[96]:


knn_1_train_score = knn_model.predict(X_train_t)
knn_train_accuracy = accuracy_score(knn_1_train_score,y_train)


# In[112]:


print('train_accuracy:',knn_train_accuracy)


# In[98]:


knn_test_score = knn_model.predict(X_test_tf)


# In[99]:


knn_test_accuracy = accuracy_score(knn_test_score,Y_test)


# In[100]:


print('test_acccuracy:',knn_test_accuracy)


# In[102]:


cmx_2=confusion_matrix(Y_test,knn_test_score)
print("\nNo. of test samples : ",len(X_test))
print("\n Confustion Matrix : \n",cmx_2)
print("\nPerfomance measures are: \n",classification_report(Y_test, knn_test_score))


# In[ ]:





# In[ ]:




