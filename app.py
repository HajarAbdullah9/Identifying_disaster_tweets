
from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
from flask import redirect
from flask import url_for
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np
import joblib
nltk.download('stopwords')
import pickle

app = Flask(__name__)
ps = PorterStemmer()

model = pickle.load(open('model.pkl', 'rb'))
count = pickle.load(open('count.pkl', 'rb'))

# Build functionalities
@app.route('/', methods=['GET'])
def home(): 

    return render_template('index.html')

def predict(text):
    tweet = re.sub('[^a-zA-Z]', ' ', text)
    tweet = tweet.lower()
    tweet = tweet.split()
    tweet = [ps.stem(word) for word in tweet if not word in stopwords.words('english')]
    tweet = ' '.join(tweet)
    tweet_vect = count.transform([tweet]).toarray()
    prediction = 'FAKE' if model.predict(tweet_vect) == 0 else 'REAL'
    return prediction
@app.route('/', methods=['POST'])
def webapp():
    text = request.form['text']
    prediction = predict(text)
    return render_template('index.html', text=text, result=prediction)
@app.route('/predict/', methods=['GET','POST'])
def api():
    text = request.args.get('text')
    prediction = predict(text)
    return jsonify(prediction=prediction)
if __name__ == "__main__":
    app.run()

