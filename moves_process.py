import pandas as pd
import chess
import chess.engine


engine = chess.engine.SimpleEngine.popen_uci("stockfish_15_win_x64_avx2\\stockfish_15_x64_avx2.exe")

moves = pd.read_csv("carlsen, magnus_moves.csv")

game_moves = (moves[moves['game_id']== "2408914d-6eff-44b0-bff7-df7f0c1774e6"])
engine.quit()


class Game:
    def __init__(self, id, moves,evaluation = []):
        self.moves = moves
        self.id = id
        self.evaluation = evaluation
        self.white_centipawn_loss = []
        self.black_centipawn_loss = []


    def evaluate(self, n = 20, t = None):
        engine = chess.engine.SimpleEngine.popen_uci("stockfish_15_win_x64_avx2\\stockfish_15_x64_avx2.exe")
        prev_centipawn = 0.3
        white = []
        black = []
        move_number = 0
        for move in self.moves:
            board = chess.Board(move)

            if move_number == 20:
                print(move)
                info = engine.analyse(board, chess.engine.Limit(depth=n, time=t))
                prev_centipawn = info["score"].white().score()
            else:
                if move_number > 20:
                    print(move)
                    info = engine.analyse(board, chess.engine.Limit(depth=n, time = t))
                    centipawn = info["score"].white().score()
                    if move_number % 2 == 0:
                        black.append(centipawn - prev_centipawn)
                    else:
                        white.append(centipawn - prev_centipawn)
                    prev_centipawn = centipawn
            move_number +=1
        engine.quit()
        self.white_centipawn_loss = white
        self.black_centipawn_loss = black

game = Game("2408914d-6eff-44b0-bff7-df7f0c1774e6", game_moves['fen'])
game.evaluate(t = 0.1)
print(game.white_centipawn_loss)
print(game.black_centipawn_loss)