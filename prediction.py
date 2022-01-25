#*************************GROUP NAME LIST************************#
# Nur Iman Kamila Binti Azharudin               1915282 
# Ani Afiqah Binti Zamrud                       1912406 
# Hilda Binti Mohd Fadzir                       1919268 
# Nur Farisya Aqilah Binti Muhamad Fadzil       1919614


#*****************************LIBRARY****************************#

from joblib import load

#***************************USER INPUT***************************#
review = input('Enter your review: ') # Get user input review.

#************************PREPROCESS INPUT************************#
# review = singleton list that contains the review that user input

review = [review.lower()] # Make it all lowercase and keep it into list of array

#************************VECTORIZE INPUT*************************#
# Vectorize the input as tf-idf array.
# Anything that we've trained will be loaded here

tfidf = load('models\\tfidf.joblib') 
vt = load('models\\vt.joblib')
review_x = tfidf.transform(review) # Review x contains tfidf that has been used
review_x = vt.transform(review_x)

#***************************PREDICTION***************************#
# Predict if the input is negative or positive.

logreg = load('models\\logreg.joblib')
review_y = logreg.predict_proba(review_x)[0] # Review y contains a list of 2 [0.6, 0.4]
                                             


if review_y[0] >= 0.5: # If probability of the review is >=0.5, model is really confident that review y is negative
    proba = review_y[0] * 100
    print(f'The review is {proba:.2f}% likely negative.') # Show the percentage of negativeness of the input
else: # Model is really confident that review y is NOT negative, hence it is positive
    proba = review_y[1] * 100 
    print(f'The review is {proba:.2f}% likely positive.') # Show the percentage of positiveness of the input