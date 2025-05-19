from flask import Flask, request

app = Flask(__name__)


@app.route('/store', methods=['POST'])
def store():
    j = request.get_json()
    print(j)


app.run()
