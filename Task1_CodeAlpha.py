#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Libraries
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import  StandardScaler


# In[2]:


data= pd.read_excel("C:\\Users\\OneDrive\\Desktop\\titanic3.xls")


# In[3]:


df= pd.DataFrame(data)


# In[4]:


df


# In[5]:


#pclass: A proxy for socio-economic status (SES)
#1st = Upper
#2nd = Middle
#3rd = Lower

#sibsp: The dataset defines family relations in this way:
#Sibling = brother, sister, stepbrother, stepsister
#Spouse = husband, wife (mistresses and fiancés were ignored)

#parch: The dataset defines family relations in this way:
#Parent = mother, father
#Child = daughter, son, stepdaughter, stepson
#Some children travelled only with a nanny, therefore parch=0 for them.


# In[6]:


df.isnull().sum()


# In[7]:


# 'embarked' and 'home.dest' are irrelevant attributes
new=df.drop(['name','cabin','embarked','boat', 'body', 'home.dest'], axis=1)
# Remove duplicates records (redundant data)
new=new.drop_duplicates()
new


# In[8]:


# Reordering the columns to make 'Survived' column (the label/ class) be the last column.
new_order= [0,2,3,4,5,6,7,1]
new=new[new.columns[new_order]]
new


# In[9]:


new.info()


# In[10]:


new.describe()


# In[11]:


# Remove null values using forward fill which takes the preceding non null value and replaces the null value  with it.
new['age'].fillna(method='ffill', inplace=True)
new['fare'].fillna(method='ffill', inplace=True)
new.describe()


# In[12]:


new['sex']=new['sex'].astype('category').cat.codes
new['ticket']=new['ticket'].astype('category').cat.codes
new


# In[13]:


corrs=new.corr(method='pearson')
# The new dataframe indicates the correlation matrix.
corrs


# In[14]:


# Creating a correlation map to show the values of the correlated and uncorrelated columns
f,ax = plt.subplots(figsize=(15, 15)) # Controls the figure size
sns.heatmap(corrs, annot=True, linewidths=.5, fmt= '.1f',ax=ax) # Annot shows the value of the correlated index
plt.title("Correlation Map")
plt.show()


# In[15]:


# Reindexing the columns and rows of the 'corrs' dataframe to a numeric representation and storing it in a new dataframe 'corrs2' for an easy access.
corrs2=corrs.set_axis((x for x in range(len(corrs))), axis='index')
corrs2=corrs2.set_axis((x for x in range(len(corrs))) , axis='columns')
corrs2


# In[16]:


# Storing the column names of the 'corrs' dataframe in a list 'ls'.
ls=corrs.columns
ls


# In[17]:


# Since the correlation matrix is a symmetric matrix, therefor we don't have to traverse all the elements.
# This loop traverse the lower triangle of the matrix only (without the main diagonal since its values = 1).
for col in range(len(corrs2)-1):
    for row in range((col+1), len(corrs2)):
# Check if the correltion is greater than or equal 0.8 (directly correlated attributes)
# or less than or equal -0.8 (inversely correlated attributes).
        if corrs2[row][col]>= 0.8 or corrs2[row][col]<= -0.8:
# If the condtition is True, remove one of the correlated attributes  from the dataset.
            new.drop(ls[col], axis=1, inplace=True)


# In[18]:


new


# In[19]:


# No attributes are removed


# In[20]:


# Saving a copy of 'new' dataset in another variable 'new2' to discretize some attribute values in it.
new2= new.copy()
# Discretization of numerical values done according to 'Pclass', 'Age', and 'Fare', binned respectively. 
new2['age']=pd.cut(x = new2['age'],
                        bins = [0,10,30,new.age.max()], 
                        labels = ['Child','Youth','Adult'])
new2['fare']=pd.cut(x= new2['fare'],
                   bins= [new2.fare.min(), np.percentile(new2.fare , 25), np.percentile(new2.fare, 50), new2.fare.max()],
                   labels= ['Low', 'Average', 'High'])
new2


# In[21]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# In[22]:


# Printing the dataset shape
print ("Dataset Shape: ", new2.shape)


# In[23]:


# Function to split the dataset
def splitdataset(df):
# Separating the target variable
     X = df.values[:, 0:len(df)-2]
     Y = df.values[:, -1]

# Splitting the dataset into train and test    
     X_train, X_test, y_train, y_test = train_test_split( 
     X, Y, test_size = 0.2, random_state = 100)
     sc= StandardScaler()
     X_train= sc.fit_transform(X_train)
     X_test= sc.fit_transform(X_test)
     
    
     return X, Y, X_train, X_test, y_train, y_test


# # Decision Tree

# In[24]:


from sklearn.tree import DecisionTreeClassifier


# In[25]:


# Function to perform training with giniIndex.
def train_using_gini(X_train, X_test, y_train):
# Creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion = "gini",
            random_state = 100,max_depth=3, min_samples_leaf=5)
# Performing training
    clf_gini.fit(X_train, y_train)
    return clf_gini


# In[26]:


# Function to perform training with entropy.
def tarin_using_entropy(X_train, X_test, y_train):
# Decision tree with entropy
      clf_entropy = DecisionTreeClassifier(
            criterion = "entropy", random_state = 100,
            max_depth = 3, min_samples_leaf = 5)
# Performing training
      clf_entropy.fit(X_train, y_train)
      return clf_entropy


# In[27]:


# Function to make predictions
def prediction(X_test, clf_object):
# Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)
    print("Predicted values:")
    print(y_pred)
    return y_pred


# In[52]:


# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
    cm=confusion_matrix(y_test, y_pred)
    print("Confusion Matrix: ",
        confusion_matrix(y_test, y_pred))
      
    print ("Accuracy : ",
    accuracy_score(y_test,y_pred)*100)
      
    print("Report : ",
    classification_report(y_test, y_pred))
    return cm    


# In[53]:


#Here's a visualization for the confusion matrix
from sklearn.svm import SVC
def plottingCM(X_train,y_train,X_test,y_test,cm):
 
    svm = SVC(kernel='rbf', random_state=0)
    svm.fit(X_train, y_train)
 
    predicted = svm.predict(X_test)
 
    #cm = confusion_matrix(y_test, predicted)
    plt.clf() # clears the entire current figure with all its axes
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia) #imshow() used to display the data as an image
    classNames = ['Negative','Positive']
    plt.title('SVM RBF Kernel Confusion Matrix - Test Data')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks,classNames) 
    plt.yticks(tick_marks, classNames)
    """xticks and yticks are responsible for determining how many labels will be fount on the x/y-axes and the name of these labels
tick_marks will determine how many elements(labels) will be put on the x/y-axes but then when we added the classnames,
the number of the elements will be replaced by the names found in the classnames array"""
    s = [['TN','FP'], ['FN', 'TP']]
 
    for i in range(2):
        for j in range(2):
            plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
    plt.show()


# In[54]:


def main(df):
      
    # Building Phase
    X, Y, X_train, X_test, y_train, y_test = splitdataset(df)
    #sc= StandardScaler()
    #X_train= sc.fit_transform(X_train)
    #X_test= sc.fit_transform(X_test)
    clf_gini = train_using_gini(X_train, X_test, y_train)
    clf_entropy = tarin_using_entropy(X_train, X_test, y_train)
      
    # Operational Phase
    print("Results Using Gini Index:")
      
    # Prediction using gini
    y_pred_gini = prediction(X_test, clf_gini)
    cm_gini=cal_accuracy(y_test, y_pred_gini)
    plottingCM(X_train,y_train,X_test,y_test,cm_gini)  
    
    print("Results Using Entropy:")
    # Prediction using entropy
    y_pred_entropy = prediction(X_test, clf_entropy)
    cm_entropy=cal_accuracy(y_test, y_pred_entropy)
    plottingCM(X_train,y_train,X_test,y_test,cm_entropy)


# In[55]:


# dataset before the dicretization of continuous attribute values
main(new)


# In[56]:


# dataset after the dicretization of continuous attribute values
new2['age']=new2['age'].astype('category').cat.codes
new2['fare']=new2['fare'].astype('category').cat.codes
main(new2)


# # KNN

# In[57]:


from sklearn.neighbors import KNeighborsClassifier


# In[58]:


# Spliting the dataset.
X, Y, X_train, X_test, y_train, y_test = splitdataset(new)


# In[59]:


# Initiating an object 'knn' from 'KNeighborsClassifier' class.
knn = KNeighborsClassifier(n_neighbors = 2)


# In[60]:


# Algorithm Implementation (Model Building)
#Step 1: Training.
knn.fit(X_train, y_train)


# In[61]:


# Getting the training data accuracy
print("Accuracy: ", knn.score(X_train, y_train))


# In[62]:


# Step 2: Classification (Prediction).
y_pred= knn.predict(X_test)
y_pred


# In[63]:


# Getting the test (predicted) data accuracy
cm_knn=cal_accuracy(y_test, y_pred)
cm_knn


# In[64]:


plottingCM(X_train,y_train,X_test,y_test,cm_knn)


# # Naive Bayes

# In[279]:


from sklearn.naive_bayes import GaussianNB


# In[280]:


# Initiating an object 'nb' from '=GaussianNB' class.
nb=GaussianNB()


# In[281]:


# Algorithm Implementation (Model Building)
#Step 1: Training.
nb.fit(X_train, y_train)


# In[282]:


# Getting the training data accuracy
print("Accuracy: ", nb.score(X_train, y_train))


# In[229]:


# Step 2: Classification (Prediction).
y_predict=nb.predict(X_test)
y_predict


# In[230]:


# Getting the test (predicted) data accuracy
cal_accuracy(y_test, y_predict)


# In[231]:


#kmeans
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans


# In[232]:


from yellowbrick.cluster import KElbowVisualizer
model = KMeans()
no_of_cols=new.shape[1]
visualizer = KElbowVisualizer(model, k=(1,no_of_cols)).fit(new)
visualizer.show()
"""The elbow method used to preidct the optimal value for the number of clusters such that when the curve starts to bend (elbow)
this is the optimal k.
Therefore the optimal k here is =2"""


# In[233]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, init='k-means++', random_state=0).fit(new)


# In[234]:


#the kmeans.labels is used to know which cluster that every data point belongs to..such that every data point will belong
#to whether cluster_0 or cluster_1
print(kmeans.labels_ ) 
print(len(kmeans.labels_)) #since the dataframe has 1284 rows therefore the labels will consist of 1284 rows
unique_labels=set(kmeans.labels_) #{0,1}


# In[235]:


kmeans.inertia_


# In[236]:


#the maximum number of iterations done by the kmeans
kmeans.n_iter_


# In[237]:


#this is used to get centroids for the 3 clusters that we have
#the ouptut array will contain k=2 elements where each each element is an array that contains no_of_cols=8 elements
centers=kmeans.cluster_centers_ 
#here we put it in the form of a dataframe to be more obvious and readable 
new_centers=pd.DataFrame(centers)
new_centers


# In[238]:


#this will count how many objects (data points) belong to the 2 clusters
from collections import Counter
Counter(kmeans.labels_)


# In[239]:


#we add a new column in the dataframe called "CLUSTER" to show which cluster does each object belong to
new['cluster']=kmeans.labels_
new


# In[240]:


from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
 
#Load Data
data = new
pca = PCA(2)
 
""" Transform the data into to a new coordinate system such that the greatest variance 
by some scalar projection of the data comes to lie on the first coordinate """
df = pca.fit_transform(data)
 
#Import KMeans module
from sklearn.cluster import KMeans
 
#Initialize the class object
kmeans = KMeans(n_clusters= 2)
 
#predict the labels of clusters.
label = kmeans.fit_predict(df)
 
#Getting unique labels
u_labels = np.unique(label)
 
#plotting the results:
for i in u_labels:
    plt.scatter(df[label == i , 0] , df[label == i , 1] , label = i)
plt.legend()
plt.show()


# In[ ]:




