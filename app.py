import json

from flask import Flask, request
import flask_cors as cors
from self_check import run_self_check

app = Flask(__name__)
cors.CORS(app)

@app.route('/')
def ping():
    return 'pong'


@app.route('/selfcheck', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS', 'HEAD'])
def self_check():
    if request.method != 'POST':
        return 'wrong request method'
    data = request.get_data()
    data = json.loads(data)
    resp_json = run_self_check(data)
    return resp_json


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=True)
