from flask import Flask, request, jsonify

app = Flask(__name__)

coords = [0,0]

@app.route("/")
def index():
    global coords
    # print(f"Latitude : {request.args['lat']} and longitude : {request.args['lon']}")
    coords = [request.args['lat'], request.args['lon']]

    return jsonify(request.data)

@app.route("/send")
def sendCoord():
    return jsonify(coords)

if __name__ == "__main__":
    app.run("0.0.0.0", 5000)

