from flask import Flask, request, jsonify
import rnn_predict

app = Flask(__name__)


@app.route('/')
def index():
    return "Welcome to python world"


@app.route('/label', methods=['POST'])
def predict_label():
    data = request.get_json()
    res = rnn_predict.predict_label(data['title'], data['content'])

    print('res: {}'.format(res))
    return jsonify({
        'label': res
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)
