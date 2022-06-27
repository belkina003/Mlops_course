import pickle

from flask import Flask, request, jsonify

with open('model.bin','rb') as f_in:
    (dv, model) = pickle.load(f_in)


def predict(features):
    X = dv.transform(features)
    preds = model.predict(X)
    return preds.mean()

app = Flask('duration-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()
    pred = predict(ride)

    result = {
        'duration': pred
    }
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)