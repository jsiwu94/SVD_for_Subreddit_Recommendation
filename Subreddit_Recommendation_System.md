# Subreddit Recommendation System using SVD

![bot](https://github.com/jsiwu94/SVD_for_Subreddit_Recommendation/blob/master/redditbot_wordcloud.png?style=centerme)

```python
import numpy as np
import pandas as pd
import math as mt
import csv
from pandas import DataFrame,Series,read_csv
import scipy
import scipy.sparse as sp
from sparsesvd import sparsesvd        #used for matrix factorization
from scipy.sparse import csc_matrix    #used for sparse matrix
from scipy.sparse.linalg import *      #used for matrix multiplication
from scipy.linalg import sqrtm
from nltk.tokenize import TreebankWordTokenizer
```



```python
reddit_df.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>username</th>
      <th>subreddit</th>
      <th>utc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>kabanossi</td>
      <td>photoshopbattles</td>
      <td>1.482748e+09</td>
    </tr>
    <tr>
      <th>1</th>
      <td>kabanossi</td>
      <td>GetMotivated</td>
      <td>1.482748e+09</td>
    </tr>
    <tr>
      <th>2</th>
      <td>kabanossi</td>
      <td>vmware</td>
      <td>1.482748e+09</td>
    </tr>
    <tr>
      <th>3</th>
      <td>kabanossi</td>
      <td>carporn</td>
      <td>1.482748e+09</td>
    </tr>
    <tr>
      <th>4</th>
      <td>kabanossi</td>
      <td>DIY</td>
      <td>1.482747e+09</td>
    </tr>
  </tbody>
</table>
</div>




```python
user = reddit_df.username.unique()
subreddit = reddit_df.subreddit.unique()
print('unique reddittor:',len(user)+1)
print('unique subreddit:',len(subreddit)+1)
print('total data entry:',reddit_df.shape)
```

    unique reddittor: 15000
    unique subreddit: 29281
    total data entry: (9391244, 3)



```python
print("Are there null values from our API dataset?  \n" + str(reddit_df.isnull().any()))
```

    Are there null values from our API dataset?  
    username     False
    subreddit    False
    utc          False
    dtype: bool


# Evaluating our SVD Model - with Test & Train by Sampling 500 users


```python
sample_username = list(reddit_df.username.unique())[300:800]
sample_df = reddit_df[reddit_df.username.isin(sample_username)]

users = list(sample_df.username.unique())
subreddits = list(sample_df.subreddit.unique())
```


```python
subs_freq = sample_df.groupby(['subreddit']
                                      , as_index=False).agg({'username': 'count'}).sort_values(by=['username']
                                      , ascending=False).reset_index(drop=True).rename(columns={'username':'username_count'})
subs_freq['cummulative_pct'] = subs_freq.username_count.cumsum()/subs_freq.username_count.sum()*100

latent_fac = subs_freq.subreddit[subs_freq.cummulative_pct <= 65].count()
contribution_pcts = round(subs_freq.cummulative_pct[len(subs_freq.subreddit[subs_freq.cummulative_pct <= 65])-1],1)

print("Top", latent_fac ,"subreddits contribute a total of"
      , contribution_pcts,"%", "to the total subreddits in the dataset")


```

    Top 134 subreddits contribute a total of 64.9 % to the total subreddits in the dataset



```python
data =sample_df.groupby(['username','subreddit']).agg({'subreddit':'count',
                                                                 'utc':'max'}).\
              rename(columns={'subreddit':'submission_freq','utc':'most_recent_timestamp'}).reset_index()
data.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>username</th>
      <th>subreddit</th>
      <th>submission_freq</th>
      <th>most_recent_timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-SA-HatfulOfHollow</td>
      <td>news</td>
      <td>1</td>
      <td>1.482761e+09</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-SA-HatfulOfHollow</td>
      <td>reddevils</td>
      <td>1</td>
      <td>1.482742e+09</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-SA-HatfulOfHollow</td>
      <td>soccer</td>
      <td>1</td>
      <td>1.482771e+09</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-SA-HatfulOfHollow</td>
      <td>worldnews</td>
      <td>11</td>
      <td>1.476293e+09</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-_-_-_-otalp-_-_-_-</td>
      <td>Android</td>
      <td>3</td>
      <td>1.475605e+09</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-_-_-_-otalp-_-_-_-</td>
      <td>AskAnthropology</td>
      <td>2</td>
      <td>1.480134e+09</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-_-_-_-otalp-_-_-_-</td>
      <td>AskReddit</td>
      <td>2</td>
      <td>1.482744e+09</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-_-_-_-otalp-_-_-_-</td>
      <td>BlackPeopleTwitter</td>
      <td>6</td>
      <td>1.482560e+09</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-_-_-_-otalp-_-_-_-</td>
      <td>CrazyIdeas</td>
      <td>1</td>
      <td>1.480079e+09</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-_-_-_-otalp-_-_-_-</td>
      <td>DC_Cinematic</td>
      <td>2</td>
      <td>1.476638e+09</td>
    </tr>
  </tbody>
</table>
</div>




```python
user_sum = data.groupby(['username'], as_index=False).agg({'submission_freq':'sum'})
temp = pd.merge(left = data, right = user_sum, how='left', left_on='username',right_on='username').\
                rename(columns={'submission_freq_y':'user_sum',
                               'submission_freq_x':'submission_freq'})
data['user_implicit_rating'] = temp.submission_freq/temp.user_sum
data.drop(columns=['submission_freq'], inplace=True)
data = pd.concat([data.iloc[:,:2],data.iloc[:,-1:],data['most_recent_timestamp']], axis=1)
data.dropna(inplace = True)
```


```python
data.shape
```




    (32018, 4)



# Splitting Train and Test Dataset based on Timestamp (utc)

- Calculating implicit rating using number of submissions per subreddit.
- Ideally need data for upvotes.


```python
users = data['username'].unique() #list of all users
subs = data['subreddit'].unique() #list of all movies

test = pd.DataFrame(columns=data.columns)
train = pd.DataFrame(columns=data.columns)
test_ratio = 0.2 #fraction of data to be used as test set.
temp1 = data[data.username.isin(users)]
for u in users:
    n = len(temp1)
    test_size = int(test_ratio*n)

temp1 = temp1.sort_values('most_recent_timestamp').reset_index()
temp1.drop('index', axis=1, inplace=True)
    
dummy_test = temp1.ix[n-1-test_size :]
dummy_train = temp1.ix[: n-2-test_size]
    
test = pd.concat([test, dummy_test])
train = pd.concat([train, dummy_train])
```

    /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/ipykernel_launcher.py:15: FutureWarning: 
    .ix is deprecated. Please use
    .loc for label based indexing or
    .iloc for positional indexing
    
    See the documentation here:
    http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#ix-indexer-is-deprecated
      from ipykernel import kernelapp as app
    /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/ipykernel_launcher.py:16: FutureWarning: 
    .ix is deprecated. Please use
    .loc for label based indexing or
    .iloc for positional indexing
    
    See the documentation here:
    http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#ix-indexer-is-deprecated
      app.launch_new_instance()



```python
print("""Train Data for User "-_-_-_-otalp-_-_-_-"        :""")
print(train[train.username == '-_-_-_-otalp-_-_-_-'].iloc[:,:3])
print(" ")
print("""Test Data for User "-_-_-_-otalp-_-_-_-"        :""")
print(test[test.username == '-_-_-_-otalp-_-_-_-'].iloc[:,:3])
```

    Train Data for User "-_-_-_-otalp-_-_-_-"        :
                      username             subreddit  user_implicit_rating
    11220  -_-_-_-otalp-_-_-_-             worldnews              0.001017
    11222  -_-_-_-otalp-_-_-_-            OCOCTATIAT              0.001017
    11365  -_-_-_-otalp-_-_-_-           nottheonion              0.001017
    11405  -_-_-_-otalp-_-_-_-     millionairemakers              0.001017
    11600  -_-_-_-otalp-_-_-_-          changemyview              0.001017
    11650  -_-_-_-otalp-_-_-_-                  news              0.001017
    12684  -_-_-_-otalp-_-_-_-               Android              0.003052
    13535  -_-_-_-otalp-_-_-_-               chomsky              0.002035
    13634  -_-_-_-otalp-_-_-_-          DC_Cinematic              0.002035
    14440  -_-_-_-otalp-_-_-_-        TheoryOfReddit              0.002035
    14453  -_-_-_-otalp-_-_-_-         UpliftingNews              0.001017
    14629  -_-_-_-otalp-_-_-_-          the_meltdown              0.001017
    15198  -_-_-_-otalp-_-_-_-               pokemon              0.001017
    15316  -_-_-_-otalp-_-_-_-             australia              0.009156
    15339  -_-_-_-otalp-_-_-_-  Political_Revolution              0.005086
    15684  -_-_-_-otalp-_-_-_-                 funny              0.004069
    15685  -_-_-_-otalp-_-_-_-                 Jokes              0.001017
    16391  -_-_-_-otalp-_-_-_-                  gifs              0.001017
    16633  -_-_-_-otalp-_-_-_-              counting              0.001017
    17434  -_-_-_-otalp-_-_-_-                iphone              0.004069
    19305  -_-_-_-otalp-_-_-_-            CrazyIdeas              0.001017
    19452  -_-_-_-otalp-_-_-_-            badhistory              0.001017
    19464  -_-_-_-otalp-_-_-_-       AskAnthropology              0.002035
    21413  -_-_-_-otalp-_-_-_-                 otalp              0.001017
    23433  -_-_-_-otalp-_-_-_-       depressedrobots              0.001017
    23808  -_-_-_-otalp-_-_-_-       EnoughTrumpSpam              0.053917
     
    Test Data for User "-_-_-_-otalp-_-_-_-"        :
                      username           subreddit  user_implicit_rating
    25662  -_-_-_-otalp-_-_-_-              me_irl              0.001017
    25708  -_-_-_-otalp-_-_-_-           socialism              0.002035
    26122  -_-_-_-otalp-_-_-_-           spacemacs              0.001017
    26438  -_-_-_-otalp-_-_-_-           radiohead              0.008138
    26816  -_-_-_-otalp-_-_-_-               meirl              0.001017
    27038  -_-_-_-otalp-_-_-_-               Music              0.005086
    27570  -_-_-_-otalp-_-_-_-          indieheads              0.001017
    28582  -_-_-_-otalp-_-_-_-      Showerthoughts              0.003052
    28839  -_-_-_-otalp-_-_-_-              movies              0.014242
    29181  -_-_-_-otalp-_-_-_-            politics              0.060020
    29186  -_-_-_-otalp-_-_-_-  BlackPeopleTwitter              0.006104
    29432  -_-_-_-otalp-_-_-_-           dagandred              0.010173
    29917  -_-_-_-otalp-_-_-_-         LiverpoolFC              0.379451
    29974  -_-_-_-otalp-_-_-_-            Fuck2016              0.018311
    30335  -_-_-_-otalp-_-_-_-            bidenbro              0.001017
    30379  -_-_-_-otalp-_-_-_-               apple              0.005086
    30955  -_-_-_-otalp-_-_-_-             atheism              0.001017
    30956  -_-_-_-otalp-_-_-_-           AskReddit              0.002035
    30969  -_-_-_-otalp-_-_-_-          botsrights              0.027467
    31486  -_-_-_-otalp-_-_-_-              soccer              0.348932


### Transforming the Dataframe into Utility Matrix for SVD Computation Later


```python
userList = data.ix[:,0].tolist()
itemList = data.ix[:,1].tolist()
valueList = data.ix[:,2].tolist()
users = list(set(data.ix[:,0]))
items = list(set(data.ix[:,1]))
users_index = {users[i]: i for i in range(len(users))}
pd_dict = {item: [np.nan for i in range(len(users))] for item in items}
for i in range(0,len(data)):
    item = itemList[i]
    user = userList[i]
    value = valueList[i]
    pd_dict[item][users_index[user]] = value
X = pd.DataFrame(pd_dict)
X.index = users
```

    /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/ipykernel_launcher.py:1: FutureWarning: 
    .ix is deprecated. Please use
    .loc for label based indexing or
    .iloc for positional indexing
    
    See the documentation here:
    http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#ix-indexer-is-deprecated
      """Entry point for launching an IPython kernel.
    /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/ipykernel_launcher.py:2: FutureWarning: 
    .ix is deprecated. Please use
    .loc for label based indexing or
    .iloc for positional indexing
    
    See the documentation here:
    http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#ix-indexer-is-deprecated
      
    /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/ipykernel_launcher.py:3: FutureWarning: 
    .ix is deprecated. Please use
    .loc for label based indexing or
    .iloc for positional indexing
    
    See the documentation here:
    http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#ix-indexer-is-deprecated
      This is separate from the ipykernel package so we can avoid doing imports until
    /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/ipykernel_launcher.py:4: FutureWarning: 
    .ix is deprecated. Please use
    .loc for label based indexing or
    .iloc for positional indexing
    
    See the documentation here:
    http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#ix-indexer-is-deprecated
      after removing the cwd from sys.path.
    /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/ipykernel_launcher.py:5: FutureWarning: 
    .ix is deprecated. Please use
    .loc for label based indexing or
    .iloc for positional indexing
    
    See the documentation here:
    http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#ix-indexer-is-deprecated
      """



```python
X[X.index == '-_-_-_-otalp-_-_-_-']['soccer']
```




    -_-_-_-otalp-_-_-_-    0.348932
    Name: soccer, dtype: float64




```python
def svd(train, k):
    utilMat = np.array(train)
    # the nan or unavailable entries are masked
    mask = np.isnan(utilMat)
    masked_arr = np.ma.masked_array(utilMat, mask)
    item_means = np.mean(masked_arr, axis=0)
    # nan entries will replaced by the average rating for each item
    utilMat = masked_arr.filled(item_means)
    x = np.tile(item_means, (utilMat.shape[0],1))
    # we remove the per item average from all entries.
    # the above mentioned nan entries will be essentially zero now
    utilMat = utilMat - x
    # The magic happens here. U and V are user and item features
    U, s, V=np.linalg.svd(utilMat, full_matrices=False)
    s=np.diag(s)
    # we take only the k most significant features
    s=s[0:k,0:k]
    U=U[:,0:k]
    V=V[0:k,:]
    s_root=sqrtm(s)
    Usk=np.dot(U,s_root)
    skV=np.dot(s_root,V)
    UsV = np.dot(Usk, skV)
    UsV = UsV + x
    print("svd done")
    return UsV
```


```python
def mse(true, pred):
    # this will be used towards the end
    x = true - pred
    return sum([xi*xi for xi in x])/len(x)

def mae(true, pred):
    # this will be used towards the end
    x = abs(true - pred)
    return sum([xi for xi in x])/len(x)


# to test the performance over a different number of features
no_of_features = [134]

svdout = svd(X, k=134)
pred = [] #to store the predicted ratings
    
for _,row in test.iterrows():
        user = row['username']
        item = row['subreddit']
        u_index = users_index[user]
        if item in items_index:
            i_index = items_index[item]
            pred_rating = svdout[u_index, i_index]
        else:
            pred_rating = np.mean(svdout[u_index, :])
        pred.append(pred_rating)
print(mse(test['user_implicit_rating'], pred))
print(mae(test['user_implicit_rating'], pred))

```

    svd done
    0.0001907806366327251
    0.005219784277697061


## Preparing the dataset for our SVD Recommendation Demo


```python
top_subreddit = reddit_df.groupby(['subreddit']
                                      , as_index=False).agg({'username': 'count'}).sort_values(by=['username']
                                      , ascending=False).reset_index(drop=True).rename(columns={'username':'username_count'})
top_subreddit['cummulative_pct'] = top_subreddit.username_count.cumsum()/top_subreddit.username_count.sum()*100
top_subreddit.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subreddit</th>
      <th>username_count</th>
      <th>cummulative_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AskReddit</td>
      <td>683932</td>
      <td>7.282656</td>
    </tr>
    <tr>
      <th>1</th>
      <td>politics</td>
      <td>260215</td>
      <td>10.053482</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The_Donald</td>
      <td>146480</td>
      <td>11.613232</td>
    </tr>
    <tr>
      <th>3</th>
      <td>nfl</td>
      <td>122088</td>
      <td>12.913252</td>
    </tr>
    <tr>
      <th>4</th>
      <td>worldnews</td>
      <td>109187</td>
      <td>14.075899</td>
    </tr>
  </tbody>
</table>
</div>




```python
user = reddit_df.username.unique()
subreddit = reddit_df.subreddit.unique()
doc_df = reddit_df.groupby('username')['subreddit'].apply(lambda x: "%s" % ' '.join(x)).reset_index()
doc_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>username</th>
      <th>subreddit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>--ANUSTART-</td>
      <td>Testosterone Testosterone Testosterone Testost...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>--Sko--</td>
      <td>DestinyTheGame DestinyTheGame DestinyTheGame D...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>--UNKN0WN--</td>
      <td>AceAttorney AceAttorney AceAttorney AceAttorne...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>--harley--quinn--</td>
      <td>LGBTeens Patriots asktransgender Patriots Patr...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-A-p-r-i-l-</td>
      <td>tdi tdi tdi AskReddit tdi tdi tdi tdi tdi tdi ...</td>
    </tr>
  </tbody>
</table>
</div>




```python
tokenizer = TreebankWordTokenizer()
document = doc_df.iloc[:, 1]
document = document.apply(lambda row: tokenizer.tokenize(row))
document.head()
```




    0    [Testosterone, Testosterone, Testosterone, Tes...
    1    [DestinyTheGame, DestinyTheGame, DestinyTheGam...
    2    [AceAttorney, AceAttorney, AceAttorney, AceAtt...
    3    [LGBTeens, Patriots, asktransgender, Patriots,...
    4    [tdi, tdi, tdi, AskReddit, tdi, tdi, tdi, tdi,...
    Name: subreddit, dtype: object



## Creating User-Subreddit Matrix
Using CSC Matrix to Handle highly sparse matrix. To view normally, use : user_subreddit_matrix.todense()


```python
corpus_of_subs = []
for subreddits in subreddit:
    corpus_of_subs.append(subreddits)


voc2id = dict(zip(corpus_of_subs, range(len(corpus_of_subs))))
rows, cols, vals = [], [], []
for r, d in enumerate(document):
    for e in d:
        if voc2id.get(e) is not None:
            rows.append(r)
            cols.append(voc2id[e])
            vals.append(1)
user_subreddit_matrix = csc_matrix((vals, (rows, cols)), dtype=np.float32)
print((user_subreddit_matrix.shape))
```

    (14999, 29280)



```python
def computeSVD(user_subreddit_matrix, no_of_latent_factors):
    
    """Compute the SVD of the given matrix.
    :user_subreddit_matrix: a numeric matrix
    :no_of_latent_factors : numeric scalar value
    
    :U  : User to concept matrix 
    :S  : Strength of the concepts matrix
    :Vt : Subreddit to concept matrix
    """
    U, s, Vt = sparsesvd(user_subreddit_matrix, no_of_latent_factors)
    
    dim = (len(s), len(s))
    S = np.zeros(dim, dtype=np.float32)
    for i in range(0, len(s)):
        S[i,i] = mt.sqrt(s[i])

    U = csc_matrix(np.transpose(U), dtype=np.float32)
    S = csc_matrix(S, dtype=np.float32)
    Vt = csc_matrix(Vt, dtype=np.float32)

    return U, S, Vt
```


```python
#Compute estimated recommendations for the given user
def computeEstimatedRecommendation(U, S, Vt, uTest):
    """Compute the recommendation for the given user.
    
    :U     : User to concept matrix 
    :S     : Strength of the concepts matrix
    :Vt    : Subreddit to concept matrix
    :uTest : Index of the user for which the recommendation has to be made
    
    :recom : List of recommendations made to the user
    """
 
    #constants defining the dimensions of the estimated rating matrix
    MAX_PID = len(subreddit)
    MAX_UID = len(user)
    
    rightTerm = S*Vt 

    EstimatedRecommendation = np.zeros(shape=(MAX_UID, MAX_PID), dtype=np.float16)
    for userTest in uTest:
        prod = U[userTest, :]*rightTerm
        # Converting the vector to dense format in order to get the indices 
        # of the movies with the best estimated ratings 
        
        EstimatedRecommendation[userTest, :] = prod.todense()
        recom = (-EstimatedRecommendation[userTest, :]).argsort()[:293]
    return recom
```


```python
n_latent_fact = top_subreddit.subreddit[top_subreddit.cummulative_pct <= 65].count()
contribution_pct = round(top_subreddit.cummulative_pct[len(subreddit[top_subreddit.cummulative_pct <= 65])-1],1)

print("Top", n_latent_fact ,"subreddits contribute a total of"
      , contribution_pct,"%", "to the total subreddits in the dataset")

```

    Top 293 subreddits contribute a total of 65.0 % to the total subreddits in the dataset


## Recommendation Demo 1


```python
no_of_latent_factors = top_subreddit.subreddit[top_subreddit.cummulative_pct <= 65].count() #293
no_of_recommendations_for_each_user = 5
uTest = [np.where(user == 'CarnationsPls')[0][0]]
U, S, Vt = computeSVD(user_subreddit_matrix, no_of_latent_factors)
```


```python
print("------------------------------------------------------------------------------------\n")
print("Redditor: %s\n" % user[uTest[0]])
print("------------------------------------------------------------------------------------\n")
print("User Subreddit History - \n")

##Getting users subs history where the vals in the matrix != 0
previous_subredit_history = subreddit[np.where(user_subreddit_matrix[uTest[0],:].todense().T != 0)[0]]
previous_subredit_history
for previous_subredits in previous_subredit_history:
     print(previous_subredits)
print("\n------------------------------------------------------------------------------------\n")
```

    ------------------------------------------------------------------------------------
    
    Redditor: CarnationsPls
    
    ------------------------------------------------------------------------------------
    
    User Subreddit History - 
    
    sports
    gaming
    gifs
    AskReddit
    fo4
    todayilearned
    TheLastAirbender
    realrule34
    
    ------------------------------------------------------------------------------------
    



```python
#Get the top 5 subreddit recommendations for test user
recommended_items = computeEstimatedRecommendation(U, S, Vt, uTest)
final_recommendation = []
for r in subreddit[recommended_items]:
    ##Making sure the subreddits aren't from what they already viewed before (for Novelty)
    if r not in previous_subredit_history:
        final_recommendation.append(r)
        if len(final_recommendation) == no_of_recommendations_for_each_user:
            break

print("------------------------------------------------------------------------------------\n")
print("Recommendation for %s : \n" % user[uTest[0]])
print("------------------------------------------------------------------------------------\n")

for recommendation in final_recommendation:
    print(recommendation)
print("------------------------------------------------------------------------------------\n")

```

    ------------------------------------------------------------------------------------
    
    Recommendation for CarnationsPls : 
    
    ------------------------------------------------------------------------------------
    
    Calgary
    Amd
    pcgaming
    techsupport
    NoMansSkyTheGame
    ------------------------------------------------------------------------------------
    


## Recommendation Demo 2


```python
no_of_latent_factors = top_subreddit.subreddit[top_subreddit.cummulative_pct <= 65].count() #293
no_of_recommendations_for_each_user = 5
uTest = [np.where(user == 'comicfan815')[0][0]]
U, S, Vt = computeSVD(user_subreddit_matrix, no_of_latent_factors)
```


```python
print("------------------------------------------------------------------------------------\n")
print("Redditor: %s\n" % user[uTest[0]])
print("------------------------------------------------------------------------------------\n")
print("User Subreddit History - \n")

##Getting users subs history where the vals in the matrix != 0
previous_subredit_history = subreddit[np.where(user_subreddit_matrix[uTest[0],:].todense().T != 0)[0]]
previous_subredit_history
for previous_subredits in previous_subredit_history:
     print(previous_subredits)
print("\n------------------------------------------------------------------------------------\n")
```

    ------------------------------------------------------------------------------------
    
    Redditor: comicfan815
    
    ------------------------------------------------------------------------------------
    
    User Subreddit History - 
    
    reactiongifs
    cringepics
    nba
    lakers
    NBA2k
    
    ------------------------------------------------------------------------------------
    



```python
#Get the top 5 subreddit recommendations for test user
recommended_items = computeEstimatedRecommendation(U, S, Vt, uTest)
final_recommendation = []
for r in subreddit[recommended_items]:
    ##Making sure the subreddits aren't from what they already viewed before (for Novelty)
    if r not in previous_subredit_history:
        final_recommendation.append(r)
        if len(final_recommendation) == no_of_recommendations_for_each_user:
            break

print("------------------------------------------------------------------------------------\n")
print("Recommendation for %s : \n" % user[uTest[0]])
print("------------------------------------------------------------------------------------\n")

for recommendation in final_recommendation:
    print(recommendation)
print("------------------------------------------------------------------------------------\n")
```

    ------------------------------------------------------------------------------------
    
    Recommendation for comicfan815 : 
    
    ------------------------------------------------------------------------------------
    
    rockets
    warriors
    bostonceltics
    sixers
    torontoraptors
    ------------------------------------------------------------------------------------
    


## The End
