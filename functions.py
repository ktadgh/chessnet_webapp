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

myclient = lichess.Client()

def white_process(evals, clocks, start_time):
    '''
    :param eval: list of integer centipawn losses
    :return: array of lists of [evaluation, centipawn loss]
    '''

    i = 0
    old_eval = 36
    old_clock = start_time
    res = []

    # iterating through centipawn losses
    for eval in evals:
        # subtracting the cpl for white's moves
        if i % 2 ==0:
            cpl = old_eval - eval
            clock_time = old_clock - clocks[i]
            res.append([cpl, clock_time])
            old_eval = eval
            old_clock = clocks[i]
            i += 1

        # adding the cpl for black's moves
        else:
            old_eval = eval
            #old_clock -= clocks[i]
            i+=1

    return numpy.array(res)
def black_process(evals, clocks, start_time):
    '''
    :param eval: list of integer centipawn losses
    :return: array of lists of [evaluation, centipawn loss]
    '''

    i = 0
    old_eval = 36
    old_clock = start_time
    res = []

    # iterating through centipawn losses
    for eval in evals:

        # subtracting the cpl for white's moves
        if i % 2 ==1:
            cpl = eval - old_eval
            clock_time = old_clock - clocks[i]
            res.append([cpl,clocks[i]])
            old_eval = eval
            old_clock = clocks[i]
            i += 1

        # adding the cpl for black's moves
        else:
            old_eval = eval
            #old_clock -= clocks[i]
            i+=1

    return numpy.array(res)

def get_times_and_evals(game_id,color, n, t): # let color be 0 for white and 1 for black
    pgn = myclient.export_by_id(game_id)
    game = chess.pgn.read_game(io.StringIO(pgn))
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

    #times = np.diff(clocks)
    if color == 0:
        return white_process(evals,clocks, 60)

    else:
        return black_process(evals, clocks, 60)

class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, no_layers):
        super(MyLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.no_layers = no_layers
        torch.manual_seed(1)
        self.lstm = nn.LSTM(input_size, hidden_size, no_layers, batch_first = True, bias = True, dropout = 0.25)
        torch.manual_seed(2)
        self.fc = nn.Linear(hidden_size,1, bias = False)
        self.final = nn.ReLU()

    def forward(self, x):

        out, _ = self.lstm(x)
        output ,lengths = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first = True)


        out = [output[e, i-1,:].unsqueeze(0)for e, i in enumerate(lengths)]
        out = torch.cat(out, dim = 0)


        out = self.fc(out)
        out = self.final(out)
        out = out[:,0]

        return out

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

def get_elo_prediction(game_id, color, n= 16,t=10, model_path='basic_model.pt', scaler_path='std_scaler.bin'):
    # defining parameters for the neural net
    input_size = 2
    hidden_size = 40
    no_layers = 4
    batch_size = 64
    model = MyLSTM(input_size, hidden_size, no_layers)
    model.load_state_dict(torch.load(model_path))
    elo_sc=load(scaler_path)
    w_eval = load('w_eval_scaler.bin')
    b_eval = load('b_eval_scaler.bin')
    value = get_times_and_evals(game_id,color, n, t)
    #print(value) # Just checking everything is ok vs Lichess analysis
    collate = MyCollator()
    eval = torch.tensor(value, dtype = torch.float32)
    transformed = w_eval.transform(eval)
    trans_tens = torch.tensor(transformed, dtype = torch.float32)
    # zipping the eval with a dummy elo, just so it works with my collate function

    data = list(zip([trans_tens], [torch.tensor([0.0])]))
    data_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, collate_fn=collate)




    for eval, elo in data_loader:
        pred_elo = model(eval).detach()
        final_elo = elo_sc.inverse_transform(pred_elo.unsqueeze(0))
    return int(final_elo[0,0])
