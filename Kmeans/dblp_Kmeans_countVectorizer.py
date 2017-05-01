
# coding: utf-8

# In[1]:

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans


# In[2]:

# dictionary of title_Id -> title_name
fp=open("title_Id.txt",'r')

titleId_titleName={}
for line in fp:
    line=line.strip().split('#')
    #print line[0], line[1]
    if len(line)!=2:
        if len(line)>2:
            key=line[-1]
            value=""
            for i in range(len(line)-1):
                value=value+line[i]
            titleId_titleName[key]=value 
    else:
        titleId_titleName[line[1]]=line[0]

fp.close()

    


# In[3]:

# dictionary of journal_Id -> journal_name
fp=open("journal_Id.txt",'r')

journalId_JournalName={}
for line in fp:
    line=line.strip().split("#")
    journalId_JournalName[line[1]]=line[0]

fp.close()


# In[4]:

# dictionary of author_Id -> author_name
fp= open("author_Id.txt",'r')

authorId_authorName={}
for line in fp:
    line = line.strip().split("#")
    authorId_authorName[line[1]]=line[0]
fp.close()


# In[5]:

fp= open("title_Journal_Author.txt",'r')

authorList=[]
titleJournalList=[]
for line in fp:
    line = line.strip().split("#")
    curr= titleId_titleName[line[0]] + " " + journalId_JournalName[line[1]]
    titleJournalList.append(curr)
    tempList=line[2].strip().split('|') #author ids list
    tempNameList=[] #authors namelist
    for i in tempList:
        tempNameList.append(authorId_authorName[i])
    authorList.append(tempNameList)
    
fp.close()


# In[6]:

#print len(authorList)
#print len(titleJournalList)
#print authorList[2], titleJournalList[2]


# In[7]:

# to find a no. of unique words
vocab={}
for line in titleJournalList:
    wordList=line.strip().split()
    for word in wordList:
        if word in vocab:
            vocab[word]=vocab[word]+1
        else:
            vocab[word]=1

print len(vocab)


# In[8]:

vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = 'english', max_features = 70000)
vec=vectorizer.fit(titleJournalList)
vectorized=vec.transform(titleJournalList)



# In[9]:

#print vectorizer


# In[10]:

#print type(vectorized)


# In[14]:


cluster_size=20
km = KMeans(n_clusters=cluster_size, init='k-means++', max_iter=100, n_init=1)
km.fit(vectorized)


# In[15]:

print("Top terms per cluster:")
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(cluster_size):
    print "Cluster %d:" % i,
    for ind in order_centroids[i, :10]:
        print ' %s' % terms[ind],
    print


# In[16]:

authorClusters={} # Cluster Id to List of authorName
for i in range(len(titleJournalList)):
    currList=[]
    currList.append(titleJournalList[i])
    data_features=vec.transform(currList)
    labels=km.predict(data_features)
    
    clusterId=labels[0]
    curAuthorsList=authorList[i] #authors list for current article
    if clusterId in authorClusters.keys():
        #fetch the existing list of authors for the clusterId
        tempList=authorClusters[clusterId]
        for j in curAuthorsList:
            tempList.append(j)
            
        #update authorsClusters with updated authors list
        authorClusters[clusterId]=tempList
        
    else: #create a new key with the new cluster id and make value list of authors for the current article
        authorClusters[clusterId]=curAuthorsList

# print clusters Id-> author ids



# In[32]:

fp=open("Kmeans_countVectorizer_output","wr")

for i in authorClusters.keys():
    outstr=str(i) + " => " + str(authorClusters[i])
    fp.write(outstr+"\n")
    fp.write("**************************************************************************************************************************"+"\n")
fp.close()


# In[33]:

get_ipython().magic(u'matplotlib inline')


# In[36]:

print (len(titleJournalList))


# In[41]:

from __future__ import print_function

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score, pairwise

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


# In[71]:

range_n_clusters = [10,20,50,70,100,130,150,170,200]


# In[72]:

valueList=[]


# In[73]:

for n_clusters in range_n_clusters:
    km = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, n_init=1,random_state=10)
    cluster_labels = km.fit_predict(vectorized)
    silhouette_avg = silhouette_score(vectorized, cluster_labels, metric='euclidean',sample_size=10000)
    valueList.append(silhouette_avg)
    print("For n_clusters =", n_clusters,"The average silhouette_score is :", silhouette_avg)


# In[75]:

valueList_15000=[]
for n_clusters in range_n_clusters:
    km = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, n_init=1,random_state=10)
    cluster_labels = km.fit_predict(vectorized)
    silhouette_avg = silhouette_score(vectorized, cluster_labels, metric='euclidean',sample_size=15000)
    valueList_15000.append(silhouette_avg)
    print("For n_clusters =", n_clusters,"The average silhouette_score is :", silhouette_avg)


# In[76]:

valueList_5000=[]
for n_clusters in range_n_clusters:
    km = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, n_init=1,random_state=10)
    cluster_labels = km.fit_predict(vectorized)
    silhouette_avg = silhouette_score(vectorized, cluster_labels, metric='euclidean',sample_size=5000)
    valueList_5000.append(silhouette_avg)
    print("For n_clusters =", n_clusters,"The average silhouette_score is :", silhouette_avg)


# In[81]:

import matplotlib.pyplot as plt

plt.axis([0,200,-0.01,0.06])
plt.xlabel('No. of Clusters')
plt.ylabel('silhouette_score')

plt.plot(range_n_clusters,valueList_5000,color='b',marker='o',label='Samples=5000')
plt.plot(range_n_clusters,valueList,color='r',marker='o',label='Samples=10000')
plt.plot(range_n_clusters,valueList_15000,color='g',marker='o',label='Samples=15000')

plt.legend()
plt.show()


# In[ ]:



