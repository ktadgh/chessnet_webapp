from flask import Flask, render_template, request
from functions import get_elo_prediction

app = Flask(__name__)

@app.route("/")
def form():
    return render_template("form2.html")


@app.route("/rating")
def rating():
    name = request.args.get("name", "0")
    color = request.args.get("color", "0")
    elo = get_elo_prediction(name, int(color))
    return render_template("rating.html", name = elo)


