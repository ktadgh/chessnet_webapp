[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
# Predicting chess results and ELO ratings based on PGN data
[webapp available here](http://webapp-external14-dev.eu-west-1.elasticbeanstalk.com)

## Neural Network
Using a Recurrent Neural network (LSTM) to predict a player's ELO rating based on Stockfish's evaluation of a single game and time spent per move. The network takes per move clock times, accuracy, and the time control as input. Results of the model are shown below, compared to a basic linear regression model based on average accuracy. The model outperforms the average accuracy model in mean squared error, and also predicts a wider range of values.

In order to ensure that the model predicts a range of values similar to the true range of ELO values, a custom loss function was implemented for training:
```
def custom_loss(outputs, elo):
    criterion = nn.MSELoss()
    mse_loss = criterion(outputs,elo)
    sample_var = ((outputs - outputs.mean())**2).mean()
    true_var = ((elo - elo.mean())**2).mean()
    var_error = (sample_var-true_var)**2
    sample_kurt = ((outputs - outputs.mean())**4).mean()
    true_kurt = ((elo - elo.mean())**4).mean()
    kurt_error = (sample_kurt-true_kurt)**2

    return mse_loss+ kurt_error/20 + var_error/20
```

The neural net used 3 hidden layers each with 24 nodes. The final hidden layer of the nerual network was fed into a final linear layer with one output. The LSTM was bidirectional, and both the forward and backward final hidden layers are used in the linear layer.


<p float="left">
<img src=https://github.com/ktadgh/chessnet_webapp/blob/master/static/baseline.png width="350" height="300" /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img src=https://github.com/ktadgh/chessnet_webapp/blob/master/static/nn.png width="350" height="300" />
 </p>


## Contents
├───Blitz Project.ipynb - *Notebook containing the analysis and logistic regression model*\
├───Database.ipynb - *The functions used to generate csvs from the pgns* \
├───NeuralNet - *Notebook containing the Neural Network* \
├───moves_process.py - *Functions to get game performance metrics*\
├───images - *Contains the graphs included above*\
├───README.md\
├───csvs\
├───pgns\
├───stockfish_15_win_x64_avx2 - *engine used for the evaluations*



