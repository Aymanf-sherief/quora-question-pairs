from flask import Flask, jsonify, request
import pickle
from scipy.sparse import hstack
from flask_cors import CORS, cross_origin
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')
model = pickle.load(open('xgboost_84.pickle', 'rb'))
vectorizer = pickle.load(open('tfidf.pickle', 'rb'))

def clean_question(text):
     words = tokenizer.tokenize(text.lower())
     return ' '.join(words)

app = Flask(__name__)
CORS(app, support_credentials=True)

@app.route('/predict', methods=['POST'])
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])
def predict():
     questions = request.get_json()
     questions = [clean_question(q) for q in questions]
     q1_features, q2_features = vectorizer.transform(questions)
     features = hstack([q1_features, q2_features])
     prediction = int(model.predict(features))

     return jsonify({'prediction': prediction})

if __name__ == '__main__':     
     app.run(port=8080, debug=True)



