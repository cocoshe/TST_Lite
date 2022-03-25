import json

import pandas as pd
from flask import Flask, request
import flask_cors as cors
from self_check import run_self_check
from train import main_, parse_args

app = Flask(__name__)
cors.CORS(app)

@app.route('/')
def ping():
    return 'pong'


@app.route('/selfcheck', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS', 'HEAD'])
def self_check():
    if request.method != 'POST':
        return 'wrong request method'
    args = parse_args()
    data = request.get_data()
    data = json.loads(data)
    data = data['data']
    data = pd.DataFrame(data)
    # print(data)
    print('data\n', data)
    resp_json = dict()
    resp_json = main_(args, data, resp_json)
    # resp_json = run_self_check(data)
    return resp_json


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
