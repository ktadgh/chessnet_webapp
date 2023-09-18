from flask import Flask, render_template, request
from functions import get_elo_prediction, get_last_board, get_winner_loser
import base64

app = Flask(__name__)

@app.route("/")
def form():
    return render_template("form2.html")


@app.route("/rating")
def rating():
    name = request.args.get("name", "0").strip('https://lichess.org/')[0:8]
    print(name)
    color = request.args.get("color", "0")

    # getting the image of the last move
    svg =get_last_board(name, color)
    svg_bytes = svg.encode('utf-8')
    svg_base64 = base64.b64encode(svg_bytes).decode()

    # getting info regarding players and result
    white, black = get_winner_loser(name, color)
    (white_player, white_color,white_points) = white
    (black_player, black_color, black_points) = black

    elo = get_elo_prediction(name, int(color))
    return render_template("rating2.html", name = elo,svg_base64=svg_base64,white_player=white_player, white_color=white_color,white_points=white_points,black_player = black_player, black_color = black_color, black_points =black_points)


