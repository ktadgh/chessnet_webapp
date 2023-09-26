# I need a python function that will tell me, from the game url and the pieces used, what somebody's predicted rating is
import lichess
import chess.pgn
import io
import chess
import numpy as np
import torch
from torch import nn
import numpy
from joblib import load
import sklearn
import chess.svg
import re

myclient = lichess.Client()


def get_init_time(string):
    t = string.split('+')
    return int(t[0])

def get_inc(string):
    t = string.split('+')
    return int(t[1])


def get_times_and_evals(game_id,color, n, t): # let color be 0 for white and 1 for black
    pgn = myclient.export_by_id(game_id)
    game = chess.pgn.read_game(io.StringIO(pgn))

    # getting starting time and increment
    increment = get_inc(game.headers['TimeControl'])
    start_time= get_init_time(game.headers['TimeControl'])

    move_number =0
    evals = []
    clocks = []
    for gme in game.mainline():
        clocks.append(gme.clock())
    engine = chess.engine.SimpleEngine.popen_uci("stockfish/stockfish-ubuntu-x86-64-modern")
    board = game.board()
    for move in game.mainline_moves():
        board.push(move)
        info = engine.analyse(board, chess.engine.Limit(depth=n, time=t))
        centipawn = info["score"].white().score(mate_score=1000)
        evals.append(centipawn)
        move_number += 1
    engine.quit()

    if color == 0:
        oppElo = float(game.headers['BlackElo'])
        return white_process(evals,clocks, start_time, start_time, increment,oppElo )

    else:
        oppElo = float(game.headers['WhiteElo'])
        return black_process(evals, clocks, start_time, start_time, increment,oppElo )

def get_winner_loser(game_id, color):
    pgn = myclient.export_by_id(game_id, literate=True)
    game = chess.pgn.read_game(io.StringIO(pgn))
    last_move = game.end()

    re1 = re.sub(r"\[.+\]",'',str(last_move))
    re2 = re.sub(r'{ ','', re1)
    re3 = re.sub(r' }', '', re2)
    re4 = re.sub(r'\d+\.', '', re3)
    re5 = re.sub(r'\D\D\d.  ','', re4)

    method_of_victory = re5.strip()
    white_player = game.headers.get("White", "Unknown White Player")
    black_player = game.headers.get("Black", "Unknown Black Player")
    white_elo = game.headers.get("WhiteElo", "?")
    black_elo = game.headers.get("BlackElo", "?")
    result = game.headers.get("Result")

    # converting the result string to points

    if result == '1-0':
        white_points = '1'
        black_points = '0'
    elif result == '0-1':
        white_points = '0'
        black_points = '1'
    else:
        white_points = '1/2'
        black_points = '1/2'


    # converting color to active colors
    if int(color) == 0:
        white_color = "#4fc94f"
        black_color = "#FFFFFF"
    else:
        black_color = "#4fc94f"
        white_color = "#FFFFFF"
    return ((white_player, white_color, white_elo, white_points), (black_player,black_color,black_elo, black_points), method_of_victory)


def get_last_board(game_id, color):
    pgn = myclient.export_by_id(game_id)
    game = chess.pgn.read_game(io.StringIO(pgn))
    board = game.board()
    for move in game.mainline_moves():
        board.push(move)
    svg = chess.svg.board(board,orientation = int(color), lastmove = move, colors = {'square light': '#c1f7c1', 'square dark':'#4fc94f', 'margin':'#000000', 'square light lastmove': '#f1f77c', 'square dark lastmove': '#f1f77c'})
    # with open(f'board.svg', 'w') as fh:
    #     fh.write(svg)
    return svg

class MyCollator(object):
    '''
    Yields a batch from a list of Items
    Args:
    test : Set True when using with test data loader. Defaults to False
    percentile : Trim sequences by this percentile
    '''

    def __call__(self, batch):
        white_data = [item[0] for item in batch]
        target = [item[1] for item in batch]
        white_lens = [i.shape[0] for i in white_data]

        white_data = torch.nn.utils.rnn.pad_sequence(white_data, batch_first=True,padding_value = 0)
        white_evals_packed = torch.nn.utils.rnn.pack_padded_sequence(white_data,batch_first = True, lengths=white_lens,enforce_sorted=False)

        target = torch.tensor(target,dtype=torch.float32)
        return [white_evals_packed, target]

#values = get_times_and_evals('KapVtqnn',0, 16, 20)

def get_elo_prediction(game_id, color, n= 16,t=10, model_path='main_model.pt', scaler_path='std_scaler.bin'):
    # defining parameters for the neural net
    input_size = 5
    hidden_size = 8
    no_layers = 2
    model = MyLSTM(input_size, hidden_size, no_layers)
    model.load_state_dict(torch.load(model_path))
    eval_scale = load('eval_scaler.bin')
    target_scale = load('target_scaler.bin')

    value = get_times_and_evals(game_id,color, n, t)

    #print(value) # Just checking everything is ok vs Lichess analysis
    collate = MyCollator()
    eval = torch.tensor(value, dtype=torch.float32)
    if color == 0:
        transformed = eval_scale.transform(eval)
    else:
        transformed = eval_scale.transform(eval)
    trans_tens = torch.tensor(transformed, dtype = torch.float32)

    # zipping the eval with a dummy elo, just so it works with my collate function
    data = list(zip([trans_tens], [torch.tensor([0.0])]))
    data_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, collate_fn=collate)


    for eval, elo in data_loader:
        pred_elo = model(eval).detach()
        final_elo = target_scale.inverse_transform(pred_elo.unsqueeze(0).unsqueeze(0))
    return int(final_elo[0,0])

### Defining the neural network
class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, no_layers):
        super(MyLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.no_layers = no_layers
        self.lstm = nn.LSTM(input_size, hidden_size, no_layers, batch_first = True, bias = True, dropout = 0, bidirectional=True)
        #self.fc = nn.Linear(hidden_size*2*no_layers,hidden_size*2*no_layers, bias = False)
        self.end = nn.Linear(hidden_size*2*no_layers,1, bias = False)
        torch.nn.init.xavier_normal_(self.lstm.weight_ih_l0, gain = 5)
        torch.nn.init.xavier_normal_(self.lstm.weight_hh_l0, gain=5)
        #self.final = nn.LeakyReLU()


    def forward(self, x):
        _, (hidden, cells) = self.lstm(x)
        #output, lengths = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        #print(f"OUT SIZE {output.size()}")
        h1 = hidden.transpose(1,0)

        l1 = h1.detach().size()[0]
        h = h1.reshape(l1,-1)
        #output ,lengths = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first = True)
        #print(output.size())

        #out = [output[e, i-1,:].unsqueeze(0) for e, i in enumerate(lengths)]
        #out = torch.cat(out, dim = 0)
        #print(f"size of edited output: {out.size()}")
        #out = self.fc(h)
        #out= self.final(out)
        out= self.end(h)
        output = torch.squeeze(out)
       #print("OUTPUT SIZE :",{out.size()})

        return output


# processing data from the csv and calculating accuracy of each move
# computing the move accuracy (as defined by Lichess)
def win_percentage(eval):
    return [50 + 50 * (2 / (1 + np.exp(-0.00368208 * centipawns)) - 1) for centipawns in eval]

def accuracy(win_prc_init, win_prc_fin):
    return 103.1668 * np.exp(-0.04354 * (win_prc_init -win_prc_fin)) - 3.1669

def black_process(evals, clocks, start_time, total_time, increment,white_elo):
    '''
    :param eval: list of integer centipawn losses
    :return: array of lists of [evaluation, centipawn loss]

    '''

    i = 0
    old_win_prc = 50
    old_clock = start_time
    res = []

    # iterating through centipawn losses
    for win_prc in win_percentage([-eval for eval in evals]):

        # subtracting the cpl for white's moves
        if i % 2 == 1:
            acc = accuracy(old_win_prc, win_prc)
            clock_time = old_clock - clocks[i]
            res.append([acc, clock_time, total_time, increment, white_elo])
            old_win_prc = win_prc
            old_clock = clocks[i]
            i += 1

        # adding the cpl for black's moves
        else:
            old_win_prc= win_prc
            # old_clock -= clocks[i]
            i += 1

    return numpy.array(res)

def white_process(evals, clocks, start_time,total_time, increment, black_elo):
    '''
    :param eval: list of integer centipawn losses
    :return: array of lists of [evaluation, centipawn loss]
    '''

    i = 0
    old_win_prc = 50
    old_clock = start_time
    res = []

    # iterating through centipawn losses
    for win_prc in win_percentage(evals):

        # subtracting the cpl for white's moves
        if i % 2 ==0:
            acc = accuracy(old_win_prc, win_prc)
            clock_time = old_clock - clocks[i]
            res.append([acc, clock_time, total_time, increment, black_elo])
            old_win_prc = win_prc
            old_clock = clocks[i]
            i += 1

        # adding the cpl for black's moves
        else:
            old_win_prc= win_prc
            #old_clock -= clocks[i]
            i+=1

    return numpy.array(res)

