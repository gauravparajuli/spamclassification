import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

#####################################################################
#                   Text Pre Processing Code Below
#####################################################################
# nltk.download('stopwords')
stopwords.words("english")

ps = PorterStemmer()


def screener(word):
    if word.isalnum():
        if word not in stopwords.words("english"):
            if word not in string.punctuation:
                return True
    return False


def transform_text(txt):
    txt = txt.lower()
    txt = nltk.word_tokenize(txt)
    txt = list(filter(screener, txt))
    txt = list(map(ps.stem, txt))
    return " ".join(txt)


#####################################################################

st.title("Email Spam Classifier")

message = st.text_area("Enter the content of the email message")

if st.button("Predict"):
    # 1 Preprocess
    transformed_message = transform_text(message)

    # 2 Vectorize
    vectorized_message = tfidf.transform([transformed_message])

    # 3 Predict
    result = model.predict(vectorized_message)[0]

    # 4 Show result
    if result == 0:
        st.header("Not Spam")
    else:
        st.header("Spam Message!")
