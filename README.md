# Traderbot Project

The inspiration for this project came from my desire to improve my skills in Machine Learning by undertaking a project that is useful to me in real life. The most immediate use case for my day to day life is to automate my trading of stocks, cryptocurrencies, and other assets. 


## Summary

The purpose of this project is to create a fully automated trading bot that live trades stocks, cryptocurrencies, and other assets. For the purposes of implementation, I have chosen to train and test this bot on Bitcoin using the global digital asset exchange (GDAX), otherwise known as coinbase pro. This project has two main components: the predictive network and the live trading network. The predictive network predicts the behavior of the asset in the near future and is used as a feature in the live trading agent. The live trading agent uses this data as well as current open, close, high, low, and volume values to make trades and hopefully turn a profit.


## Data Collection

I collected all data for this project from GDAX using the [cbpro API](https://pypi.org/project/cbpro2/). The data contained the open, close, high, low, and volume metrics for bitcoin over an entire day in five minute intervals. One of the challenges presented during data collection was addressing missing values. More popular coins like Bitcoin and Ethereum have no gaps in their price history, however less popular coins like Adam have multiple gaps. The way data is stored in GDAX makes it seem as though there is no gap save for a descrepency in timestamps. This meant a part of the data cleaning process was to identify and fill these gaps. Instead of making it a separate step, I chose to implement it as a part of the data collection process. Ordinarily, one could ask for five minute interval data spanning any amount of time for a coin. However I chose to ask for one five minute interval at a time, and should that five minute interval be empty, it would save as an empty row to the csv file. Once this process was done, I could aggregate it into a larger csv file for training and fill in any missing gaps using a forward fill to prevent data leakage.


## Predictor

The predictive network uses a transformer based model, the inspiration for which can be found [here](https://github.com/JanSchm/CapMarket/blob/master/bot_experiments)/IBM_Transformer%2BTimeEmbedding.ipynb. I chose to use a transformer based model because it is one of the more advanced memory based machine learning models available, has shown better results than the LSTM, and its data hungry nature could be easily satisfied by years of Bitcoin data for training. The predictive model had to be adjusted over multiple runs to find a balance between keeping an up to date prediction and computational efficency. While it originally only predicted five minutes in advance, now it generated a prediction for the open, low, high, close, and volume for the next calendar day. This is then adjusted to show whether or not these values will be higher or lower than the current day's values (1 and -1 respectively) to make it more digestible for the live trading agent. These values are then streamed to the live trading agent along with current prices.


## Live trading agent

The live trading agent was trained using the [tensortrade](tensortrade.org) library in conjunction with ray (ray.io). Discussion on the community discord allowed me to minimize my experimentation with feature engineering by getting a sense of what parameters yielded the best results in the long run. Thus, the optimal model uses the differential sharpe ratio as the reward function, the log differences of the open, close, high, low and volume as well as the predicted values in the stream, and looks at a historical window of 160 time units in order to make decisions. For the sake of expediency, I deployed an earlier model using net worth as the reward function, because the differential sharpe ratio yields exceedingly long training times. 


## Deployment

The bot itself is deployed as a single program that runs every five minutes from a server. The actual deployment takes advantage of the fact that the bot itself will make the same decisions on the same provided data; if the bot is run from a point in the past up to the present, and run again similarly five minutes later, it will make the same decisions as it did in the first run save for the extra time point at the end. This method can be implemented to make a live trading decision every minutes by looking at the final state of the model. To prevent the data feed from becoming too large, the agent at the second timestep is saved on every run and loaded as the first state on the next to maintain window size. 

The set up that is used is run from a linode Ubuntu server using files in the [Server](https://github.com/Gary-McC/TraderbotFinal/tree/main/Server) folder all in a single folder. The bot itself is run every five minutes by editting the server [crontab](https://techbast.com/2021/05/linux-how-to-install-and-use-crontab-on-ubuntu-server.html), and the actions of the bot are recorded in the Bot Logs.csv file. More details for running the bot can be found in the [Server](https://github.com/Gary-McC/TraderbotFinal/tree/main/Server) folder.

## Installation and Use

The bot final bot itself requires the files found in the Server Folder. In its current state, it is meant to run from a live server or local machine every five minutes. It is possible to customize the bots behavior through data collection, predictor retraining, and RL agent retraining, code for which is have supplied in the appropriate folders. For convenience, I have also supplied some basic predictive and RL models for immediate use and prototyping.
