
import pandas as pd
import seaborn as sns
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score, f1_score, precision_score,confusion_matrix, recall_score, roc_auc_score
import nltk, re, string #THIS important to get stopwords and remove it
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import model_selection

from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
import seaborn as sns
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import GridSpec
df = pd.read_csv ('C:\\Users\\LENOVO\\Desktop\\webapp\\df.csv')
df.head()

# create a corpus:
from nltk.stem import PorterStemmer
df.head(10)
corpus  = []
pstem = PorterStemmer()
for i in range(df['cleaned_tweets'].shape[0]):
    tweet = re.sub("[^a-zA-Z]", ' ', df['cleaned_tweets'][i])
    #Split the words
    tweet = tweet.split()
    #Remove stopwords then Stemming it
    tweet = [pstem.stem(word) for word in tweet if not word in set(stopwords.words('english'))]
    tweet = ' '.join(tweet) 
    #Append cleaned tweet to corpus
    corpus.append(tweet)

df = df.drop(['text'],axis=1)
print("Corpus created successfully") 
print(pd.DataFrame(corpus)[0].head(10))
rawTexData = df["cleaned_tweets"].head(10)
cleanTexData = pd.DataFrame(corpus, columns=['corpus'])
frames = [rawTexData, cleanTexData]
result = pd.concat(frames, axis=1, sort=False)
result

#Create our dictionary 
uniqueWordFrequents = {}
for tweet in corpus:
    for word in tweet.split():
        if(word in uniqueWordFrequents.keys()):
            uniqueWordFrequents[word] += 1
        else:
            uniqueWordFrequents[word] = 1
            
#Convert dictionary to dataFrame
uniqueWordFrequents = pd.DataFrame.from_dict(uniqueWordFrequents,orient='index',columns=['Word Frequent'])
uniqueWordFrequents.sort_values(by=['Word Frequent'], inplace=True, ascending=False)
uniqueWordFrequents.head(10)
uniqueWordFrequents['Word Frequent'].unique()
uniqueWordFrequents = uniqueWordFrequents[uniqueWordFrequents['Word Frequent'] >= 20]
print(uniqueWordFrequents.shape)
from sklearn.feature_extraction.text import CountVectorizer
counVec = CountVectorizer(max_features = uniqueWordFrequents.shape[0])
bagOfWords = counVec.fit_transform(corpus).toarray()



b = bagOfWords
y = df['target']

print("b shape = ",b.shape)
print("y shape = ",y.shape)

b_train , b_test , y_train , y_test = train_test_split(b,y,test_size=0.20, random_state=55, shuffle =True)
print('data splitting successfully')

multinomialNBModel = MultinomialNB(alpha=0.1)
multinomialNBModel.fit(b_train,y_train)
print("multinomialNB model run successfully")

passModel=PassiveAggressiveClassifier()
passModel.fit(b_train,y_train)
print ('Passive Regressive model run successfully')

modelsNames = [('multinomialNBModel',multinomialNBModel),
               ('PassiveAggressiveClassifier',passModel)]

from sklearn.ensemble import VotingClassifier
votingClassifier = VotingClassifier(voting = 'hard',estimators= modelsNames)
votingClassifier.fit(b_train,y_train)
print("votingClassifier model run successfully")



models = [multinomialNBModel, passModel, votingClassifier]

for model in models:
    print(type(model).__name__,' Train Score is   : ' ,model.score(b_train, y_train))
    print(type(model).__name__,' Test Score is    : ' ,model.score(b_test, y_test))
    
    y_pred = model.predict(b_test)
    print(type(model).__name__,' F1 Score is      : ' ,f1_score(y_test,y_pred))



import pickle
pickle.dump(votingClassifier , open("model.pkl", "wb"))
pickle.dump(counVec, open("count.pkl", "wb"))

