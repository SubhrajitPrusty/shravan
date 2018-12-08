from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def index():
    print(f"Latitude : {request.args['lat']} and longitude : {request.args['lon']}")
    return jsonify(request.data)

if __name__ == "__main__":
    app.run("0.0.0.0", 5000)

