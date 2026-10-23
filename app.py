from flask import Flask, render_template, request, jsonify
from chat import get_response, bot_name
from cbf import get_similarity_matrix

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    msg  = data["message"]
    response = get_response(msg)
    return jsonify(response)

@app.route("/similarity", methods=["GET"])
def similarity():
    kategori = request.args.get("kategori", "Nasi Goreng")
    matrix   = get_similarity_matrix(kategori)
    return jsonify({
        "kategori": kategori,
        "matrix"  : matrix
    })

if __name__ == "__main__":
    app.run(debug=True)