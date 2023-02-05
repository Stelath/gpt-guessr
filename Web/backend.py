from flask import Flask, request
from flask_cors import CORS
import json
import requests
import subprocess

from selProgram import run_test

app = Flask(__name__)
CORS(app, origins=["http://localhost:8000/"])

@app.route("/", methods =['GET', 'POST'])
def handle_request():
    print("Here")
    request_data = request.get_json()
    
    request_text = request_data.get("request", "")
    print(request_text)
    # Run the Python script with the request text
    output = run_test(request_text)
    print("After subprocess!")
    response_data = { "response": output }
    print(response_data)
    return json.dumps(response_data), 200

if __name__ == "__main__":
    #app.run(debug=True)
    app.run(host='0.0.0.0')