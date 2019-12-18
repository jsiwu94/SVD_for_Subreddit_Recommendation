# Getting the Data Using Reddit API (PRAW)


```python
import praw
import configparser
import random
import pandas as pd
import numpy as np
import sys
```


```python
reddit = praw.Reddit(client_id='Si81-WNFEeTb1w',
                    client_secret='ndqNnvsuiAWyi-fqvK8h-dsZqqo',
                    user_agent='subreddit_data')
```

## Initial Simple API Command Test


```python
for submission in reddit.subreddit('all').hot(limit=5):
    print(submission.title)
```

    TIL actor Robert Pattinson dealt with an obsessed fan who had been camping outside his apartment by taking her out on a dinner date. "I just complained about everything in my life and she never came back."
    Pay your fucking employees. This is some dystopian shit.
    For my birthday this year, my wife asked what kind of cake she should make... and I told her that, honestly, I've always wanted to try Peach's thank-you cake from the end of Super Mario 64.
    Poor kid...hahaha
    Every morning this dog waits for his girls to get on bus.



```python
python_title = []
time = []
num_upvotes = []
num_comments = []
upvote_ratio = []
link_flair = []
redditor = []
i=0
for submission in reddit.subreddit('learnpython').top(limit=10):
    i+=1
    python_title.append(submission.title)
    time.append(submission.created_utc)
    num_upvotes.append(submission.score)
    num_comments.append(submission.num_comments)
    upvote_ratio.append(submission.upvote_ratio)
    link_flair.append(submission.link_flair_text)
    redditor.append(submission.author)
    if i%5 == 0:
        print(f'{i} submissions completed')
```

    5 submissions completed
    10 submissions completed



```python
df = pd.DataFrame(
    {'python_title': python_title,
     'time': time,
     'num_comments': num_comments,
     'num_upvotes': num_upvotes,
     'upvote_ratio': upvote_ratio,
     'link_flair': link_flair,
     'redditor': redditor
    })
df.head(10)
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
      <th>python_title</th>
      <th>time</th>
      <th>num_comments</th>
      <th>num_upvotes</th>
      <th>upvote_ratio</th>
      <th>link_flair</th>
      <th>redditor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The online course for "Automate the Boring Stu...</td>
      <td>1.560203e+09</td>
      <td>245</td>
      <td>2079</td>
      <td>0.99</td>
      <td>None</td>
      <td>AlSweigart</td>
    </tr>
    <tr>
      <th>1</th>
      <td>I'm 100% self taught, landed my first job! My ...</td>
      <td>1.566412e+09</td>
      <td>311</td>
      <td>1837</td>
      <td>0.99</td>
      <td>None</td>
      <td>JLaurus</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Beginner's Python Cheat Sheets (updated)</td>
      <td>1.571419e+09</td>
      <td>131</td>
      <td>1406</td>
      <td>0.99</td>
      <td>None</td>
      <td>ehmatthes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>I'm super annoyed and taking it out on learnpy...</td>
      <td>1.560047e+09</td>
      <td>256</td>
      <td>1373</td>
      <td>0.97</td>
      <td>None</td>
      <td>SpergLordMcFappyPant</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Why study programming when you can just play a...</td>
      <td>1.551413e+09</td>
      <td>181</td>
      <td>1099</td>
      <td>0.97</td>
      <td>None</td>
      <td>TorroesPrime</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Automate the Boring Stuff FREE on Udemy</td>
      <td>1.572898e+09</td>
      <td>96</td>
      <td>987</td>
      <td>0.99</td>
      <td>None</td>
      <td>TechsInTheCity</td>
    </tr>
    <tr>
      <th>6</th>
      <td>"Automate the Boring Stuff with Python" Udemy ...</td>
      <td>1.575339e+09</td>
      <td>92</td>
      <td>906</td>
      <td>0.99</td>
      <td>None</td>
      <td>AlSweigart</td>
    </tr>
    <tr>
      <th>7</th>
      <td>"Automate the Boring Stuff" author refactoring...</td>
      <td>1.555450e+09</td>
      <td>67</td>
      <td>901</td>
      <td>0.99</td>
      <td>None</td>
      <td>AlSweigart</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Al Sweigart, author of the legendary 'Automate...</td>
      <td>1.549548e+09</td>
      <td>56</td>
      <td>855</td>
      <td>0.99</td>
      <td>None</td>
      <td>callmelucky</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Nothing is more liberating than learning pytho...</td>
      <td>1.566807e+09</td>
      <td>51</td>
      <td>851</td>
      <td>0.98</td>
      <td>None</td>
      <td>GeneticalTM</td>
    </tr>
  </tbody>
</table>
</div>



## Getting The Subreddit Dataset for Analysis :
- 15,000 unique redditor
- their subreddit submissions
- utc (for timestamp)


```python
r = reddit.subreddit('all').top(limit=15000)
redditor = []
subreddit_name = []
utc = []
i=0

for submission in r:
    redditor.append(submission.author)
    subreddit_name.append(submission.subreddit)
    utc.append(submission.created_utc)
    i+=1
print(f'{i} unique users retrieved')
```

    971 unique users retrieved



```python
redditor[:10]
```




    [Redditor(name='iH8myPP'),
     Redditor(name='Itsjorgehernandez'),
     Redditor(name='SrGrafo'),
     Redditor(name='SageMo'),
     Redditor(name='datbanter'),
     Redditor(name='foxeydog'),
     Redditor(name='rightcoastguy'),
     Redditor(name='System32Comics'),
     Redditor(name='Sickpupz'),
     Redditor(name='iswearidk')]




```python
subreddit_name[:10]
```




    [Subreddit(display_name='funny'),
     Subreddit(display_name='pics'),
     Subreddit(display_name='gaming'),
     Subreddit(display_name='news'),
     Subreddit(display_name='pics'),
     Subreddit(display_name='pics'),
     Subreddit(display_name='pics'),
     Subreddit(display_name='funny'),
     Subreddit(display_name='memes'),
     Subreddit(display_name='gifs')]




```python
utc[:10]
```




    [1480959674.0,
     1478651245.0,
     1563035293.0,
     1570653917.0,
     1486400800.0,
     1565569541.0,
     1565375556.0,
     1568139947.0,
     1561206210.0,
     1482425855.0]




```python
reddit = pd.DataFrame(
    {'username': redditor,
     'subreddit': subreddit_name,
     'utc': utc
    })
reddit.head(10)
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
      <th>utc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>iH8myPP</td>
      <td>funny</td>
      <td>1.480960e+09</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Itsjorgehernandez</td>
      <td>pics</td>
      <td>1.478651e+09</td>
    </tr>
    <tr>
      <th>2</th>
      <td>SrGrafo</td>
      <td>gaming</td>
      <td>1.563035e+09</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SageMo</td>
      <td>news</td>
      <td>1.570654e+09</td>
    </tr>
    <tr>
      <th>4</th>
      <td>datbanter</td>
      <td>pics</td>
      <td>1.486401e+09</td>
    </tr>
    <tr>
      <th>5</th>
      <td>foxeydog</td>
      <td>pics</td>
      <td>1.565570e+09</td>
    </tr>
    <tr>
      <th>6</th>
      <td>rightcoastguy</td>
      <td>pics</td>
      <td>1.565376e+09</td>
    </tr>
    <tr>
      <th>7</th>
      <td>System32Comics</td>
      <td>funny</td>
      <td>1.568140e+09</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Sickpupz</td>
      <td>memes</td>
      <td>1.561206e+09</td>
    </tr>
    <tr>
      <th>9</th>
      <td>iswearidk</td>
      <td>gifs</td>
      <td>1.482426e+09</td>
    </tr>
  </tbody>
</table>
</div>




```python
reddit.to_csv('../reddit_praw.csv')
```
