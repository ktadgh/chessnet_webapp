[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
# Predicting chess results and ELO ratings based on PGN data
[webapp available here](http://webapp-external14-dev.eu-west-1.elasticbeanstalk.com)

## Neural Network
Using a Recurrent Neural network (LSTM) to predict a player's ELO rating based on Stockfish's evaluation of a single game and time spent per move. The network takes per move clock times, accuracy, and the time control as input. Results of the model are shown below, compared to a basic linear regression model based on average accuracy. The model appears to be outperforming the average centipawn loss model in mean squared error, and also predicts a wider range of values. 

<p float="left">
<img src=https://github.com/ktadgh/chessnet_webapp/blob/master/baseline.png width="350" height="300" /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img src=https://github.com/ktadgh/chessnet_webapp/blob/master/nn.png width="350" height="300" />
 </p>

## Blitz Project
Using logistic regression to predict the result of a chess game based on the players' ELO ratings and other factors.
The regression model returns the predicted probability of a win, loss and draw, and overall can predict the expected points more accurately
than the ELO alone.

<p float="left">
<img src=https://github.com/ktadgh/chessnet/blob/main/images/ELO_acc.png width="350" height="300" /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img src=https://github.com/ktadgh/chessnet/blob/main/images/Model_acc.png width="350" height="300" />
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



