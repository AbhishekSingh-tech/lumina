import json
from flask import Flask, jsonify, request
from prediction import predict
from pipe_instance import PipeProvider

application = Flask(__name__)
pipe_object = PipeProvider()
pipe = pipe_object.load_pipe()

@application.route('/')
@application.route('/status')
def status():
    return jsonify({'status': 'ok'})


@application.route('/predictions', methods=['POST'])
def create_prediction():
    data = request.data or '{}'
    body = json.loads(data)
    return jsonify(predict(body, pipe))
