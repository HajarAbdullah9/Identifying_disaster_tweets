# Identifying_disaster_tweets
This project aims to apply knowledge and abilities in the fields of data analysis and machine learning 
in order to develop a classification model based on supervised learning that can accurately determine if
a piece of information (tweet text) refers to a real disaster or not. 
analysis.py:
For this analysis i followed 2 separated paths of vectorizing methods, 1st was TFIDF vectorizer, and 2nd was BOW (bag of words) vectorizer. 
For the TFIDF path i applied the MultiNominal Naive Bayse model and   Passive Aggressive model, to predict the related tweets.
On other hand, for the BOW path i applied thes same two models with VotingClassifier model. Finally i perform the test for the analysis depending on TFIDF
vectorizer trough entering tweets under one variable as an array, then do the test. For the other path i preferred to change the way for testing so i created
shiny-r to let the user entering his own tweet, what ever the tweet will be, user will got accurate result depending on the previous (bag of words) vectorizer. Thank you
