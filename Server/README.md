# Server Deployment

This folder contains all necessary files for deploying the bot. Simply copy all files inside the folder and run either the full bot or the paper trading bot. Full details below.

## Full_Trader_Bot.py

Python script for actual live trading with the bot. Uses the Cbpro API to trade through coinbase pro. To run, set up an account (pro.coinbase.com) and get an API key(https://cryptopro.app/help/automatic-import/coinbase-pro-api-key/). Type the relevant API information in line 680 of the code, update your relevant starting wallet information in Starting_Wallet.csv, and run the program. Records of the trades will be recording in Bot Logs.csv. Ensure Transformer+TimeEmbedding+BTC-USD-24-hour-all.hdf5 and the RL Agent Model folder are all in the same folder.

## Full_Trader_Bot_Paper.py

Effectively the same program as Full_Trader_Bot.py, except it trades with fake money instead of real money. Records of the fake trades will be recorded in Bot Logs.csv

## Starting Wallet.csv

Contains the values for your starting wallet in the program. Both trader bot programs constantly update this as they run, however you will have to fill this in with your own starting values for initialization. Ensure it is in the same folder as the trader bot programs.

## Bot Logs.csv

Contains the trade logs for the trading bots. Ensure it is in the same folder as the trader bot programs.

## Transformer+TimeEmbedding+BTC-USD-24-hour-all.hdf5

Model for the predictive model used in the traderbot. Ensure it is in the same folder as the trader bot programs. 

The actual predictive model uses a transformer based model to predict tomorrow's Open, Close, High, and Low prices and feed them as features to the datafeed. To reduce both computational load and avoid any compounding accuracy issues, these predictive features are sent as -1 if the value lowers in 24 hours, and 1 if the value increases in 24 hours. It is important to update this model every two weeks or so to ensure the predictions stay up to date with current asset behavior. Programs for doing that can be found in the Predictive Model folder from the main page.

## RL Agent Model folder

Contains the relevant files for instantiating the RL Agent for trading decisions.  Ensure it is in the same folder as the trader bot programs.

Current model is a basic model using Net Worth as a reward function, and was implemented using the example code from the tensortrade docs (https://www.tensortrade.org/en/latest/tutorials/ray.html). The current model uses a basic Net Worth reward function, with well as the log-differences of the price history on five minute ticks and the predictive features as inputs, and was trained on one year of price data. The input features were chosen after some feature engineering experimentation and discussion with other users from the tensortrade community. There is certainly room for training a better model than the one provided in the example here. The pre-trained one here is simply offered for the sake of fast implementation for prototyping.
