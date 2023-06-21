import numpy as np 
import pandas as pd
import seaborn as sns
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk, re, string 
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score,confusion_matrix, recall_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import KFold 
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
df = pd.read_csv ('C:\\Users\\LENOVO\\Desktop\\slides\\3rd semester\\CP\\train.csv')
df.shape
df.head()
df.isnull().sum()
df = df.drop(['id','keyword','location'],axis=1)
df.isnull().sum()
sns.set_style("dark")
VCdf=df['target'].value_counts().to_frame()
sns.countplot(x='target', data= df)
plt.title('Real or Un-Real Disaster Tweet')
df['target'].value_counts().plot.pie(autopct='%1.2f%%')

df['length']=df['text'].apply(len)
df.head()
df.length.describe()
df_1 = df[df['target']==1]
df_0 = df[df['target']==0]
display("Random sample of disaster tweets:",df[df.target==1].text.sample(3).to_frame())
display("Random sample of non disaster tweets:",df[df.target==0].text.sample(3).to_frame())

stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)


#removing unnccesary stopwords 
def remove_stopwords(text):
    final_text=[]
    for i in text.split():
        if i.strip().lower() not in stop:
            final_text.append(i.strip())
    return " ".join(final_text)
df_1['text']=df_1['text'].apply(remove_stopwords)
df_0['text']=df_0['text'].apply(remove_stopwords)
df['text'].apply(remove_stopwords)

# creat wordscloud for both of subset datesets
from wordcloud import WordCloud
plt.figure(figsize=(20,15))
wc = WordCloud(max_words =1000, width=1600, height=800).generate(" ".join(df_1.text))
plt.imshow(wc, interpolation="bilinear")
from wordcloud import WordCloud
plt.figure(figsize=(20,15))
wc = WordCloud(max_words=1000, width=1600, height=800).generate(" ".join(df_0.text))
plt.imshow(wc, interpolation="bilinear")


from nltk.stem import WordNetLemmatizer
lemma= WordNetLemmatizer()
stop = stopwords.words('English')
def clean_tweets(text):
    text = text.lower()
    words= nltk.word_tokenize(text)
    words = ' '.join ([lemma.lemmatize(word) for word in words 
                     if word not in (stop)])
    text=''.join(words)
    text = re.sub('[^a-z]',' ',text)
    return text
df['cleaned_tweets'] = df['text'].apply(clean_tweets)
df.head()

#more Exeplorar Analysis visualisationn
# Creating a new feature for the visualization.

df['Character Count'] = df['cleaned_tweets'].apply(lambda x: len(str(x)))

from matplotlib.ticker import MaxNLocator
from matplotlib import gridspec
def plot_dist3(dfrm, feature, title):
    # Creating a customized chart. and giving in figsize and everything.
    fig = plt.figure(constrained_layout=True, figsize=(18, 8))
    # Creating a grid of 3 cols and 3 rows.
    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)

    # Customizing the histogram grid.
    ax1 = fig.add_subplot(grid[0, :2])
    # Set the title.
    ax1.set_title('Histogram')
    # plot the histogram.
    sns.distplot(dfrm.loc[:, feature],
                 hist=True,
                 kde=True,
                 ax=ax1,
                 color='#e74c3c')
    ax1.set(ylabel='Frequency')
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=20))
    plt.suptitle(f'{title}', fontsize=24)
plot_dist3(df[df['target'] == 0], 'Character Count',
           'Characters Per "Non Disaster" Tweet')
plot_dist3(df[df['target'] == 1], 'Character Count',
           'Characters Per " Disaster" Tweet')

#word count:
def plot_word_number_histogram(textno, textye):
    
    """A function for comparing word counts"""

    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(18, 6), sharey=True)
    sns.distplot(textno.str.split().map(lambda x: len(x)), ax=axes[0], color='#e74c3c')
    sns.distplot(textye.str.split().map(lambda x: len(x)), ax=axes[1], color='#e74c3c')
    
    axes[0].set_xlabel('Word Count')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Non Disaster Tweets')
    axes[1].set_xlabel('Word Count')
    axes[1].set_title('Disaster Tweets')
    
    fig.suptitle('Words Per Tweet', fontsize=24, va='baseline')
    
    fig.tight_layout()
plot_word_number_histogram(df[df['target'] == 0]['text'],
                           df[df['target'] == 1]['text'])


# Displaying most common words.
lis = [
    df[df['target'] == 0]['cleaned_tweets'],
    df[df['target'] == 1]['cleaned_tweets']
]

from collections import Counter, defaultdict

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
axes = axes.flatten()

for i, j in zip(lis, axes):
        new = i.str.split()
        new = new.values.tolist()
        corpus = [word for i in new for word in i]
        
        counter = Counter(corpus)
        most = counter.most_common()
        x, y = [], []
        for word, count in most[:50]:
            if (word not in stop):
                x.append(word)
                y.append(count)

        sns.barplot(x=y, y=x, palette='plasma', ax=j)
        axes[0].set_title('Non Disaster Tweets')

axes[1].set_title('Disaster Tweets')
axes[0].set_xlabel('Count')
axes[0].set_ylabel('Word')
axes[1].set_xlabel('Count')
axes[1].set_ylabel('Word')

fig.suptitle('Most Common Unigrams', fontsize=24, va='baseline')
plt.tight_layout()

from sklearn.feature_extraction.text import CountVectorizer
def ngrams(n, title):
    """A Function to plot most common ngrams"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    axes = axes.flatten()
    for i, j in zip(lis, axes):

        new = i.str.split()
        new = new.values.tolist()
        corpus = [word for i in new for word in i]

        def _get_top_ngram(corpus, n=None):
            #getting top ngrams
            vec = CountVectorizer(ngram_range=(n, n),
                                  max_df=0.9,
                                  stop_words='english').fit(corpus)
            bag_of_words = vec.transform(corpus)
            sum_words = bag_of_words.sum(axis=0)
            words_freq = [(word, sum_words[0, idx])
                          for word, idx in vec.vocabulary_.items()]
            words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
            return words_freq[:15]

        top_n_bigrams = _get_top_ngram(i, n)[:15]
        x, y = map(list, zip(*top_n_bigrams))
        sns.barplot(x=y, y=x, palette='plasma', ax=j)
        
        axes[0].set_title('Non Disaster Tweets')
        axes[1].set_title('Disaster Tweets')
        axes[0].set_xlabel('Count')
        axes[0].set_ylabel('Words')
        axes[1].set_xlabel('Count')
        axes[1].set_ylabel('Words')
        fig.suptitle(title, fontsize=24, va='baseline')
        plt.tight_layout()
        
ngrams(1, 'Most Common Ngrams')
ngrams(2, 'Most Common Bigrams')
ngrams(3, 'Most Common Trigrams')

# start with BWO Vectorizer
# create a corpus:
from nltk.stem import PorterStemmer
df.head(10)

corpus2  = []
pstem = PorterStemmer()
for i in range(df['cleaned_tweets'].shape[0]):
    tweet = df['cleaned_tweets'][i]
    tweet= tweet.split()
    tweet = [pstem.stem(word) for word in tweet if not word in set(stopwords.words('english'))]
    tweet = ' '.join(tweet)
    #Append cleaned tweet to corpus
    corpus2.append(tweet)
print("Corpus created successfully") 


print(pd.DataFrame(corpus2)[0].head(10))
rawTexData = df["text"].head(10)
cleanTexData = pd.DataFrame(corpus2, columns=['text after cleaning']).head(10)
frames = [rawTexData, cleanTexData]
result = pd.concat(frames, axis=1, sort=False)
result
#Create our dictionary 
uniqueWordFrequents = {}
for tweet in corpus2:
    for word in tweet.split():
        if(word in uniqueWordFrequents.keys()):
            uniqueWordFrequents[word] += 1
        else:
            uniqueWordFrequents[word] = 1
            
#Convert dictionary to dataFrame
uniqueWordFrequents = pd.DataFrame.from_dict(uniqueWordFrequents,orient='index',
                                             columns=['Word Frequent'])
uniqueWordFrequents.sort_values(by=['Word Frequent'], inplace=True, ascending=False)
uniqueWordFrequents.head(10)
uniqueWordFrequents['Word Frequent'].unique()
uniqueWordFrequents = uniqueWordFrequents[uniqueWordFrequents['Word Frequent'] >= 20]
print(uniqueWordFrequents.shape)
from sklearn.feature_extraction.text import CountVectorizer
counVec = CountVectorizer(max_features = uniqueWordFrequents.shape[0])
bagOfWords = counVec.fit_transform(corpus2).toarray()

b = bagOfWords
y = df['target']
print("b shape = ",b.shape)
print("y shape = ",y.shape)

b_train , b_test , y_train , y_test = train_test_split(b,y,test_size=0.20, 
                            random_state=55, shuffle =True)
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

#evaluation Details
models = [multinomialNBModel, passModel, votingClassifier]

for model in models:
    print(type(model).__name__,' Train Score is   : ' ,model.score(b_train, y_train))
    print(type(model).__name__,' Test Score is    : ' ,model.score(b_test, y_test))
    
    y_pred = model.predict(b_test)
    print(type(model).__name__,' F1 Score is      : ' ,f1_score(y_test,y_pred))
    print('--------------------------------------------------------------------------')
    
# Now i will apply TFD-DF on the choosen models (Multinominal Naive Bayse and passive aggressive classifiers):

x= df.cleaned_tweets
y = df.target
#train test split
# The stratify parameter asks whether you want to retain the same proportion of classes in the train and test sets that are found in the entire original dataset. For example, if there are 100 observations in the entire original dataset of which 80 are class a and 20 are class b and you set stratify = True , with a
# the random_state parameter is used for initializing the internal random number generator, which will decide the splitting of data into train and test indices in your case. If random_state is None or np. random, then a randomly-initialized RandomState object is returned.
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.20,
                                                    stratify=y, random_state=0)

# now i will vectorize the text data to bigram and trigram
#Bigram tfidf 
tfidf_vectorizer_2 = TfidfVectorizer(stop_words='english',max_df=0.8,
                                     ngram_range=(1,2))
tfidf_train_2 = tfidf_vectorizer_2.fit_transform(x_train)
tfidf_test_2 = tfidf_vectorizer_2.transform(x_test)

#Trigram_tfidf
tfidf_vectorizer_3 = TfidfVectorizer(stop_words='english',max_df=0.8, ngram_range= (1,3))
tfidf_train_3 = tfidf_vectorizer_3.fit_transform(x_train)
tfidf_test_3 = tfidf_vectorizer_3.transform(x_test) 

#Build machine learning model
# 1- Multinomunal Naive Bayes for Bigram&Trigram:
mnb_tf_bigram=MultinomialNB()
mnb_tf_bigram.fit(tfidf_train_2, y_train)
mnb_tf_trigram=MultinomialNB()
mnb_tf_trigram.fit(tfidf_train_3,y_train)

#passive aggressive classifire for Bigram&Trigram:
pass_tf_bigram=PassiveAggressiveClassifier()
pass_tf_bigram.fit(tfidf_train_2,y_train)
pass_tf_trigram=PassiveAggressiveClassifier()
pass_tf_trigram.fit(tfidf_train_3,y_train)

#after we fitted the classifieres models on the training set we should now vross validate
kfold=model_selection.KFold(n_splits=10)
scoring='accuracy'

acc_mnb2 = cross_val_score(estimator= mnb_tf_bigram, X= tfidf_train_2, y=y_train, 
                           cv= kfold, scoring= scoring)
acc_mnb3 = cross_val_score(estimator= mnb_tf_trigram, X= tfidf_train_3,y= y_train, 
                           cv= kfold, scoring= scoring)

acc_pass2= cross_val_score(estimator = pass_tf_bigram, X= tfidf_train_2, y= y_train,
                           cv= kfold, scoring=scoring)
acc_pass3 = cross_val_score(estimator= pass_tf_trigram, X= tfidf_train_3, y= y_train, 
                            cv= kfold, scoring=scoring)



import gensim
from gensim.models import Word2Vec
from collections import defaultdict


#comparing between the moldels accuracy accros validation of 10 training splits 
crossdict = {
    'MNB_Bigram':acc_mnb2.mean(),
    'Passive_Aggressive_Bigram':acc_pass2.mean(),
    'MNB_Trigram': acc_mnb3.mean(),
    'Passive_Aggressive_Trigram':acc_pass3.mean()
}
#A Pandas DataFrame is a 2 dimensional data structure, like a 2 dimensional array, or a table with rows and columns.
cross_df = pd.DataFrame(crossdict.items(), columns=['Model','Cross_val accuracy'])
cross_df=cross_df.sort_values(by=['Cross_val accuracy'], ascending=False)
cross_df

# as we can see the passive_aggressive_bigram is winn here whith high cross validation accuracy with 0.579146
# now we will evaluate both models which are already vectorized by the 4 evaluation metrics
# i will start with mnb_bigram
pred_mnb2 = mnb_tf_bigram.predict(tfidf_test_2)
acc= accuracy_score(y_test,pred_mnb2)
prec = precision_score(y_test, pred_mnb2)
rec = recall_score(y_test,pred_mnb2)
f1 = f1_score(y_test,pred_mnb2)
roc= roc_auc_score(y_test,pred_mnb2)

model_results = pd.DataFrame([['Multinoinal Naive Bayes- Bigram', acc,prec,rec,f1,roc]], 
            columns = ['Model','Accuracy', 'Precision','Sensitivity', 'F1-score','ROC'])
model_results

# now we will compare with other models
pred_mnb3= mnb_tf_trigram.predict(tfidf_test_3)     
pred_pass2 = pass_tf_bigram.predict(tfidf_test_2)
pred_pass3= pass_tf_trigram.predict(tfidf_test_3)

models = {
    'Multinominal Naive Bayes-Trigram' : pred_mnb3,
    'Passive Aggressive-Bigram' : pred_pass2,
    'passive Aggressive-Trigram' : pred_pass3
}

models= pd.DataFrame(models)
for column in models:
        acc= accuracy_score(y_test,models[column])
        prec= precision_score(y_test,models[column])
        rec= recall_score (y_test,models[column])
        f1= f1_score (y_test,models[column])
        roc= roc_auc_score (y_test,models[column])
        results = pd.DataFrame ([[column, acc,prec,rec,f1,roc]], 
                                columns = ['Model','Accuracy', 'Precision','Sensitivity', 'F1-score','ROC'])
        model_results = model_results._append (results, ignore_index=True)
model_results

# now i will define function it will highlight highest evaluation scores
def highlight_max (s):
    np.object = object    
    if s.dtype == object:
        is_max= [False for _ in range(s.shape[0])]
    else:
        is_max = s == s.max()
   #print(is_max)
    return ['background: lightgreen' if cell else '' for cell in is_max]

model_results.style.apply(highlight_max)

def most_informative_feature_for_binary_classification(vectorizer, classifier, n=100):
      class_labels = classifier.classes_
      feature_names = vectorizer.get_feature_names_out()
      topn_class1 = sorted(zip(classifier.coef_[0], feature_names))[:n]
      topn_class2 = sorted(zip(classifier.coef_[0], feature_names))[-n:]
      for coef, feat in topn_class1:
        print(class_labels[0], coef, feat)
      print()
      for coef, feat in reversed(topn_class2):
        print(class_labels[1], coef, feat)
most_informative_feature_for_binary_classification(tfidf_vectorizer_3, pass_tf_trigram,n=10)
        
sentences = [
        "Hi  i love reading in quite place",
        "there is earthquake in our area",
        "what is her nationality",
        "there is a strong storm in our city, please every one take care",
        "there is a fire inside this building"
    ]
    
tfidf_trigram = tfidf_vectorizer_3.transform(sentences)
predictions = pass_tf_trigram.predict(tfidf_trigram)
for text, label in zip(sentences, predictions):
        if label==1:
                target="Disaster Tweet"
                print("text:", text, "\nClass:", target)
                print()
        else:
            target="Normal Tweet"
            print("text:", text, "\nClass:", target)
            print()


