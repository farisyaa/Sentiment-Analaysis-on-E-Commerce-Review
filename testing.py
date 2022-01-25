#*************************GROUP NAME LIST************************#
# Nur Iman Kamila Binti Azharudin               1915282 
# Ani Afiqah Binti Zamrud                       1912406 
# Hilda Binti Mohd Fadzir                       1919268 
# Nur Farisya Aqilah Binti Muhamad Fadzil       1919614


#*****************************LIBRARY****************************#

from joblib import load
import numpy as np

#***************************USER INPUT***************************#

print('Noted: The number must be in range 0-999!')
index_test = int(input('Select a review number to be tested: '))


#************************LOAD/CLEAN DATA*************************#

reviews = []
labels = []
with open('test.txt') as test_file:
    lines = test_file.read().splitlines()
    for line in lines:
        tokens = line.split(' ')
        label = tokens[0].replace('__label__', '')  # Remove __label__.
        text = ' '.join(tokens[1:]).lower()  # GREAT or Great -> great 
        reviews.append(text)
        labels.append(label)

review = [reviews[index_test]]

#************************VECTORIZE INPUT*************************#
# Vectorize the input as tf-idf array.
# Anything that we've trained will be loaded here

tfidf = load('models\\tfidf.joblib')
vt = load('models\\vt.joblib')

#***************************PREDICTION***************************#

# Check if the review is in the range
if index_test <= 999 and index_test > 0:
    review_x = tfidf.transform(review) # Review x contains tfidf that has been used
    review_x = vt.transform(review_x)

    # Predict if the input is negative or positive.
    logreg = load('models\\logreg.joblib')
    review_y = logreg.predict_proba(review_x)[0] # Review y contains a list of 2 [0.6, 0.4]
    

    print('The review: ', review[0])

    if review_y[0] >= 0.5: # If probability of the review is >=0.5, model is really confident that review y is negative
        proba = review_y[0] * 100
        print(f'The review is {proba:.2f}% likely negative.') # Show the percentage of negativeness of the input
    else: # Model is really confident that review y is NOT negative, hence it is positive
        proba = review_y[1] * 100
        print(f'The review is {proba:.2f}% likely positive.') # Show the percentage of positiveness of the input

# Print if the review is out of range
else:
    print('The number is out of range!')

    for review in reviews:
        review_x = tfidf.transform([review])
        review_x = vt.transform(review_x)
        
        logreg = load('models\\logreg.joblib')
        review_y = logreg.predict_proba(review_x)[0]

        if review_y[0] >= 0.5:
            proba = review_y[0] * 100
            print(f'The review is {proba:.2f}% likely negative.')
        else:
            proba = review_y[1] * 100
            print(f'The review is {proba:.2f}% likely positive.')