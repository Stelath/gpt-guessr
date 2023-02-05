from flask import Flask, request
from flask_cors import CORS
import json
import requests
import subprocess

app = Flask(__name__)
CORS(app, origins=["http://localhost:8000/"])

@app.route("/", methods =['GET', 'POST'])
def handle_request():
    print("Here")
    request_data = request.get_json()
    
    request_text = request_data.get("request", "")
    print(request_text)
    # Run the Python script with the request text
    output = subprocess.run(["py", "selProgram.py", request_text], capture_output=True, text=True)
    print("After subprocess!")
    response_data = { "response": output.stdout }
    print(response_data)
    return json.dumps(response_data), 200

if __name__ == "__main__":
    #app.run(debug=True)
    app.run(host='0.0.0.0')