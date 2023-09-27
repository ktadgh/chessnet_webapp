from flask import Flask, render_template, request
from functions import get_elo_prediction, get_last_board, get_winner_loser,get_key_moves
import base64

app = Flask(__name__)

@app.route("/")
def form():
    return render_template("form2.html")


@app.route("/rating")
def rating():
    name = request.args.get("name", "0").strip('https://lichess.org/')[0:8]
    color = request.args.get("color", "0")
    #error handling:
    if name == '':
        name = '1587xZMq'

    # getting the image of the last move
    try:
        svg =get_last_board(name, color)
    except:
        return render_template("form3.html")

    svg_bytes = svg.encode('utf-8')
    svg_base64 = base64.b64encode(svg_bytes).decode()

    # getting info regarding players and result
    white, black, method_of_victory = get_winner_loser(name, color)
    (white_player, white_color,white_elo,white_points) = white
    (black_player, black_color, black_elo,black_points) = black

    elo = get_elo_prediction(name, int(color))

    key_moves, index = get_key_moves(name, int(color))

    return render_template("rating2.html",index = index, name = elo,svg_base64=svg_base64,white_player=white_player, white_color=white_color,white_points=white_points,black_player = black_player, black_color = black_color, black_points =black_points, white_elo=white_elo, black_elo=black_elo, method_of_victory=method_of_victory, key_moves = key_moves)


