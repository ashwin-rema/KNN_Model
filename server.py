import json

import joblib
from flask import Flask, request

app = Flask(__name__)

# Load the .pkl model
model = joblib.load('knn_classifier_model.pkl')
encoder = joblib.load('encoder.joblib')
y_train = ['Q123', 'C100', 'T67894', 'T67894', 'Q123', 'Q123', 'C100', 'T67895', 'T67897', 'N123456', 'N123456', 'Q123',
           'T67897', 'T67899', 'C100', 'T67897', 'T67899', 'N123789', 'T67895', 'Q123', 'T67899', 'C100', 'T67890',
           'C100', 'T67899', 'T67895', 'N123456', 'N123789', 'N123456', 'T67895', 'Q123', 'T67894', 'T67899', 'T67895',
           'T67895', 'T67897', 'T67895', 'Q123', 'T67890', 'N123456', 'T67897', 'Q456', 'T67890', 'T67895', 'Q456',
           'T67897', 'N123456', 'N123789', 'T67895', 'N123456', 'Q123', 'T67899', 'T67897', 'N123456', 'N123789',
           'C100', 'Q456', 'N123789', 'C100', 'T67897', 'Q123', 'N123456', 'N123456', 'T67899', 'N123456', 'N123789',
           'Q123', 'Q456', 'Q123', 'N123456', 'T67890', 'Q456', 'C100', 'Q456', 'T67890', 'T67897', 'T67899', 'T67899',
           'T67897', 'T67894', 'T67890', 'T67894', 'C100', 'T67890', 'T67895', 'Q456', 'C100', 'N123789', 'N123456',
           'T67897', 'T67890', 'N123789', 'N123456', 'T67899', 'T67894', 'T67897', 'T67890', 'Q456', 'T67890', 'Q123',
           'Q456', 'Q123', 'N123456', 'Q456', 'T67897', 'Q456', 'T67894', 'C100', 'T67894', 'Q123', 'N123789', 'T67895',
           'T67899', 'N123456', 'N123456', 'Q456', 'T67895', 'C100', 'T67894', 'C100', 'Q123', 'Q456', 'N123456',
           'T67899', 'T67895', 'T67897', 'T67899', 'Q123', 'T67894', 'T67899', 'T67899', 'Q123', 'Q456', 'N123456',
           'N123456', 'T67899', 'T67894', 'Q123', 'T67897', 'T67890', 'Q123', 'T67890', 'T67899', 'N123456', 'C100',
           'N123789', 'T67894', 'Q123', 'C100', 'T67897', 'T67890', 'T67897', 'T67894', 'T67899', 'T67895', 'T67899',
           'Q123', 'Q123', 'T67894', 'T67899', 'T67894', 'N123789', 'N123456', 'Q123', 'N123789', 'Q123', 'T67894',
           'T67894', 'T67897', 'T67894', 'T67890', 'T67895', 'Q456', 'T67895', 'T67895', 'T67897', 'T67890', 'T67895',
           'N123456', 'Q456', 'Q123', 'Q123', 'T67895', 'T67890', 'T67895', 'T67897', 'N123789', 'C100', 'T67899',
           'T67895', 'T67890', 'T67899', 'Q456', 'N123789', 'T67897', 'N123456', 'C100', 'Q123', 'Q123', 'T67890',
           'N123456', 'Q123', 'Q456', 'Q123', 'Q456', 'Q123', 'T67899', 'T67899', 'T67899', 'T67897', 'Q123', 'N123456',
           'T67895', 'T67899', 'T67899', 'T67894', 'T67895', 'N123456', 'C100', 'T67894', 'Q123', 'Q456', 'Q456',
           'N123789', 'N123789', 'Q123', 'Q123', 'T67890', 'T67894', 'Q456', 'T67899', 'T67897', 'C100', 'T67897',
           'Q456', 'T67890', 'C100', 'T67894', 'Q456', 'T67894']


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    final_list = []
    final_list.append(data['riskTolerance'])
    final_list.append(data['incomeCategory'])
    final_list.append(data['lengthOfInvestment'])
    new_data_encoded = encoder.transform([final_list])
    # prediction = model.predict(new_data_encoded)
    k = 5
    _, neighbor_indices = model.kneighbors(new_data_encoded, n_neighbors=k)
    neighbor_labels = [y_train[i] for i in neighbor_indices[0]]
    from collections import Counter
    label_counts = Counter(neighbor_labels)
    top_5_predictions = [label for label, count in label_counts.most_common(5)]

    return json.dumps(top_5_predictions)

    # return jsonify({'prediction_1': top_5_predictions[0],
    #                 'prediction_2': top_5_predictions[1],
    #                 'prediction_3': top_5_predictions[2],
    #                 'prediction_4': top_5_predictions[3]})


@app.route('/')
def index():
    return 'Hello, World!'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
