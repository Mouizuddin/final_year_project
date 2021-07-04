import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# ml lib 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string
#
import streamlit as st


# ### Inserting fake and real dataset

# In[2]:


df_fake = pd.read_csv('cvs/Fake.csv')
df_true = pd.read_csv('cvs/True.csv')


# In[3]:

st.title('Reading head of the files')
st.write(df_fake.head(5))


# In[4]:
st.write(df_true.head(5))

# ### Inserting a column called "class" for fake and real news dataset to categories fake and true news.

df_fake["class"] = 0
df_true["class"] = 1
df_true.head()


# ### Removing last 10 rows from both the dataset, for manual testing  

# In[6]:


df_fake.shape, df_true.shape


# In[7]:


df_fake_manual_testing = df_fake.tail(10)
# for i in range(23480,23470,-1):
#     df_fake.drop([i], axis = 0, inplace = True)
df_true_manual_testing = df_true.tail(10)
# for i in range(21416,21406,-1):
#     df_true.drop([i], axis = 0, inplace = True)


# In[8]:


df_fake.shape, df_true.shape


# ## Merging the manual testing dataframe in single dataset and save it in a csv file

# In[9]:


df_fake_manual_testing["class"] = 0
df_true_manual_testing["class"] = 1


# In[10]:


df_fake_manual_testing.head(10)


# In[11]:


df_true_manual_testing.head(10)


# ## creating a manual file

# In[12]:

st.title('creating a manual file')
df_manual_testing = pd.concat([df_fake_manual_testing,df_true_manual_testing], axis = 0)
df_manual_testing.to_csv("totest.csv")


# ## Merging the main fake and true dataframe

# In[13]:


df_marge = pd.concat([df_fake, df_true], axis =0 )

df_marge.head(10)


# In[14]:


st.write(df_marge.columns)


# #### "title",  "subject" and "date" columns is not required for detecting the fake news, so I am going to drop the columns.

# In[15]:


df = df_marge.drop(["title", "subject","date"], axis = 1)


# In[16]:


df.isnull().sum()


# ## Randomly shuffling the dataframe
st.title('Randomly shuffling the dataframe ')
# In[17]:


df = df.sample(frac = 1)
st.write(df.head())

df.reset_index(inplace = True)
df.drop(["index"], axis = 1, inplace = True)

df.columns


df.head()


# ### Creating a function to convert the text in lowercase, remove the extra space, special chr., ulr and links.

# In[22]:


def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text


# In[23]:


df["text"] = df["text"].apply(wordopt)


# #### Defining dependent and independent variable as x and y

# In[24]:


x = df["text"]
y = df["class"]


# #### Splitting the dataset into training set and testing set. 
st.title('Splitting the dataset into training set and testing set')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)


# #### Convert text to vectors

# In[26]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[27]:


vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)


# ### 1. Logistic Regression

# In[28]:
st.title('Logistic Regression')

from sklearn.linear_model import LogisticRegression


# In[29]:


LR = LogisticRegression()
st.write(LR.fit(xv_train,y_train))


# In[30]:


pred_lr=LR.predict(xv_test)

st.write(pred_lr)
# In[31]:


st.write(LR.score(xv_test, y_test))


# In[32]:


print(classification_report(y_test, pred_lr))
st.write(classification_report(y_test, pred_lr))

# ### 2. Decision Tree Classification

# In[33]:
st.title(' Decision Tree Classification')

from sklearn.tree import DecisionTreeClassifier


# In[34]:


DT = DecisionTreeClassifier()
st.write(DT.fit(xv_train, y_train))


# In[35]:


pred_dt = DT.predict(xv_test)
st.write(pred_dt)

# In[36]:


st.write(DT.score(xv_test, y_test))


# In[37]:


print(classification_report(y_test, pred_dt))


# ### 3. Gradient Boosting Classifier

# In[38]:
st.title('Gradient Boosting Classifier')

from sklearn.ensemble import GradientBoostingClassifier


# In[39]:


GBC = GradientBoostingClassifier(random_state=0)
st.write(GBC.fit(xv_train, y_train))


# In[40]:


pred_gbc = GBC.predict(xv_test)
st.write(pred_gbc)

# In[41]:


st.write(GBC.score(xv_test, y_test))

st.write(classification_report(y_test, pred_gbc))
print()

st.title('Random Forest Classifier')
from sklearn.ensemble import RandomForestClassifier


RFC = RandomForestClassifier(random_state=0)
RFC.fit(xv_train, y_train)

pred_rfc = RFC.predict(xv_test)

st.write(pred_rfc)

st.write(RFC.score(xv_test, y_test))
print(classification_report(y_test, pred_rfc))

# In[48]:


def output_lable(n):
    if n == 0:
        return ' "Fake News - (FALSE)" '
    elif n == 1:
        return ' "Not A Fake News - (TRUE)" '
    
def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GBC = GBC.predict(new_xv_test)
    pred_RFC = RFC.predict(new_xv_test)

    return print("\n\nLinear Regressior Prediction: {} \nDecision Tree classifier Prediction: {} \nGradient Boosting Classifier Prediction: {} \nRandom Forest Classifier Prediction: {}".format(output_lable(pred_LR[0]), 
                                                                                                              output_lable(pred_DT[0]), 
                                                                                                              output_lable(pred_GBC[0]), 
                                                                                                              output_lable(pred_RFC[0])))

news = str(input())
st.title('Make Predections')
predection = st.text_area('Make Predections')
# st.title('Make Predections')
print(manual_testing(news))
st.write(predection)

# In[ ]:





# ## Time complexities 
# 

# - Random forest classfiers : O(d∗n∗log(n)) - O(log(n))
# - DTC : O(m · n2) - O(log(n^2)
# - LR : O(n)
# - O(n*logn) > O(n)
