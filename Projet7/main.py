from fastapi import FastAPI
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import uvicorn
from pydantic import BaseModel
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')

app = FastAPI()
pickle_in_class = open("log_reg.pkl", "rb")
pickle_in_tfidf = open("tfidf.pkl", "rb")
classifier = pickle.load(pickle_in_class)

tf = pickle.load(pickle_in_tfidf)


class TextInput(BaseModel):
    text: str


def tweeter(sentence):
    print(sentence)
    print(type(sentence))
    stemmer = PorterStemmer()
    tk = TweetTokenizer(preserve_case=False, reduce_len=True)
    tok_sent = tk.tokenize(str(sentence))
    stop_words = set(stopwords.words('english'))
    text = [stemmer.stem(word.lower())
            for word in tok_sent
            if word not in stop_words
            and word.isalpha()]
    sent = ""
    for word in text:
        sent += word + " "
    return sent[:-1]


@app.get("/")
def root():
    return {"Hello": "World"}


@app.post("/predict")
async def predict(text: TextInput):
    processedTweet = tweeter(text)
    processedTweet = [processedTweet]
    new_text_tfidf = tf.transform(processedTweet).toarray()

    try:
        prediction = classifier.predict(new_text_tfidf)
        print(prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "Prediction": prediction[0]
    }


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
