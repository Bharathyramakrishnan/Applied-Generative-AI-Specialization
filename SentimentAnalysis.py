import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import re

nltk.download('stopwords')
nltk.download('wordnet')

reviews = pd.read_csv("C:/D_Drive/Applied-Generative-AI-Specialization/NLP_dataset.csv",encoding='unicode_escape')
reviews = pd.DataFrame(reviews)
#print(reviews)

# reviews = [
# "Great cooler.. excellent air flow and for this price. It's so amazing and unbelievable. Just love it ƒ?‹?",
# "Best budget 2 fit cooler. Nice cooling",
# "The quality is good but the power of air is decent",
# "Very bad product it's a only a fan",
# "Ok ok product",
# "The cooler is really fantastic and provides good air flow. Highly recommended",
# "Very good product",
# "Very nice",
# "Very bad cooler",
# "Very good",
# "Beautiful product good material and perfectly working",
# "Awesome",
# "Good",
# "Wonderful product, Must buy",
# "Nice air cooler, smart cool breeze producer",
# "Awsm",
# "Nice product ???",
# "Great cooler..",
# "Nice product",
# "Good ???",
# "Very nice product ???? ???",
# "Good product",
# "Nice product with the reasonable price",
# "I like it...ƒœ‹?",
# "Very goodd",
# "Good product",
# "Good product kawaleti",
# "Very good cooler amazing beautiful designs affordable price",
# "Using since 3months. Great experience.",
# "Very good performance and nice look",
# "Product is good having strong thrust of air flow.. Must buy",
# "Very good ???",
# "Bad quality"
# ]

def preprocess_text(text):
    # Convert to string to handle non-string inputs
    text = str(text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    
    return text

# Apply Preprocess to each review
preprocessed_reviews = [preprocess_text(reviews) for review in reviews]

from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize tfidf vectorizer
tfidf_vertorizer = TfidfVectorizer()

# Fit Transforms the preprossed reviews
tfidf_matrix = tfidf_vertorizer.fit_transform(preprocessed_reviews)

# Convert TF-IDF matrix to Dataframe for visualization (optional)
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vertorizer.get_feature_names_out())

#print(tfidf_matrix)

#print(tfidf_df)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Define the sentiment labels and their corresponding numerical values
sentiment_mapping = {
    "Positive" : 1,
    "Neutral" : 0,
    "Negative" : -1
}

# Assign Numnerical lables to each row based on the sentiment
y_train = [1,1,0,-1,0,1,1,1,-1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1]

# split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(tfidf_df,y_train,test_size=0.2,random_state=42)

# initialize and train logistic regression model
logistic_regression = LogisticRegression()
logistic_regression.fit(x_train,y_train)

#Predict sentiment labels for test data 
y_pred = logistic_regression.predict(x_test)

# Ensure that the data splitting process is correct 
print(len(x_train))
print(len(x_test))

# Ensure the both training and testing sets contains samples from each class
print(y_train)
print(y_test)

#Evaluate accuracy
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)
