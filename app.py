from flask import Flask, request, jsonify
# from flask_sqlalchemy import SQLAlchemy
# from flask_marshmallow import Marshmallow
import os
import requests
from requests.structures import CaseInsensitiveDict
import json

app = Flask(__name__)

@app.route('/api/sentiment', methods=['POST'])
def sentiment():

    user = request.json['user']
    num = request.json['num']

    token = 'BEARER_TOKEN'

    # text = request.json['text']
    headers = CaseInsensitiveDict()
    # headers["Accept"] = "application/json"
    headers["Authorization"] = "Bearer "+token

    resp = requests.get('https://api.twitter.com/2/users/by/username/'+user, headers=headers).json()

    # resp2 = requests.get('https://api.twitter.com/2/users/'+resp['data']['id']+'/tweets', headers=headers).json() 
    resp2 = requests.get('https://api.twitter.com/2/users/'+resp['data']['id']+'/tweets?tweet.fields=public_metrics&'+'max_results='+num, headers=headers).json() 

    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    model = load_model('./lstm/best_model.h5')
    # model = load_model_from_hdf5('C:/Users/91961/Desktop/sentiment/lstm/best_model.h5')
    import pickle

    max_words = 5000
    max_len=50

    tokenizer = Tokenizer(num_words=max_words, lower=True, split=' ')

    with open('./lstm/preprocess.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # tokenizer.fit_on_texts(text)
    # tokenizer = Tokenizer(num_words=max_words, lower=True, split=' ')

    def predict_class(text):
        '''Function to predict sentiment class of the passed text'''
        
        sentiment_classes = ['Negative', 'Neutral', 'Positive']
        max_len=50
        # print(text)
        # Transforms text to a sequence of integers using a tokenizer object
        # tokenizer = Tokenizer()
        
        xt = tokenizer.texts_to_sequences(text)
        print(xt)
        # Pad sequences to the same length
        xt = pad_sequences(xt, padding='post', maxlen=max_len)
        print(xt)
        # Do the prediction using the loaded model
        yt = model.predict(xt).argmax(axis=1)
        print(model.predict(xt))
        # Print the predicted sentiment
        k= sentiment_classes[yt[0]]
        # print(k)
        return k

    final_dict = {}
    res_array = []

    for i in resp2['data']:
        tweet = i['text']
        analysis = predict_class([tweet])
        i['sentiment']=analysis
        # final_dict[tweet]=analysis
        # temp = final_dict
        # final_dict = {}
        # res_array.append(temp)
        # temp = {}
    print(resp2)
    # return jsonify({ "result": predict_class([text])})
    # return jsonify({ "result": res_array })
    return jsonify({ "result": resp2['data'] })

# @app.route('/api/lstm', methods=['POST'])
# def lstm():
#     text = request.json['text']
#     from tensorflow.keras.models import load_model
#     from tensorflow.keras.preprocessing.text import Tokenizer
#     from tensorflow.keras.preprocessing.sequence import pad_sequences
#     model = load_model('./lstm/best_model.h5')
#     import pickle
#     max_words = 5000
#     max_len=50
#     tokenizer = Tokenizer(num_words=max_words, lower=True, split=' ')
#     with open('./lstm/preprocess.pickle', 'rb') as handle:
#         tokenizer = pickle.load(handle)
#     # tokenizer.fit_on_texts(text)
#     # tokenizer = Tokenizer(num_words=max_words, lower=True, split=' ')
#     def predict_class(text):
#         '''Function to predict sentiment class of the passed text'''
        
#         sentiment_classes = ['Negative', 'Neutral', 'Positive']
#         max_len=50
#         # print(text)
#         # Transforms text to a sequence of integers using a tokenizer object
#         # tokenizer = Tokenizer()
        
#         xt = tokenizer.texts_to_sequences(text)
#         print(xt)
#         # Pad sequences to the same length
#         xt = pad_sequences(xt, padding='post', maxlen=max_len)
#         print(xt)
#         # Do the prediction using the loaded model
#         yt = model.predict(xt).argmax(axis=1)
#         print(model.predict(xt))
#         # Print the predicted sentiment
#         k= sentiment_classes[yt[0]]
#         # print(k)
#         return k

#     return jsonify({ "result": predict_class([text])})

if __name__ == '__main__':
    app.run(debug=True)