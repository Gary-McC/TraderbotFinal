# RL Agent

This folder contains the necessary files for training the RL Agent. The RL agent uses the tensortrade (tensortrade.org) library combined with ray(ray.io) to train and deploy the model. Because tensortrade does not natively support live trading, the agent makes "live" decisions by reinstantiating itself every five minutes, and making a decision from a point in the past (currently five days ago) to the present,and makes its present decision by looking at the final action. This works because the agent makes the same decisions when given the same data, so slowly shifting the timeframe for making decisions does not introduce significant variability in the model. 
The only piece of error handling that needs to be taken into account when doing this is that the agent cannot start in a state with 0 USD,or a "bought in" state without failing. To avoid this, the starting wallet is updated and saved to a csv file every run, taking the first point (excluding time=0) where there is no bought in state to save as the starting point for the next run. If point is any point greater than t=1, the starting point will remain fixed until the window size once again reaches five days. This five day window size was specifically chosen so the bought can have the opportunity to exit a buy within the time window to ensure there is a "sold" state somewhere in the window.

Discussion on the community discord allowed me to minimize my experimentation with feature engineering by getting a sense of what parameters yielded the best results in the long run. Thus, the optimal model uses the differential sharpe ratio as the reward function, the log differences of the open, close, high, low and volume as well as the predicted values in the stream, and looks at a historical window of 160 time units in order to make decisions. For the sake of expediency, I deployed an earlier model using net worth as the reward function, because the differential sharpe ratio yields exceedingly long training times.

# Ray RL Mod.ipynb

This notebook contains the example code used to train the initial agent on one year of data. The training was done in a notebook as opposed to a script for experimentation purposes. The default reward scheme is currently set to Simple Profit for ease of use on the users end, however using Diff_Sharpe yields better results with significantly longer training times.

To run, simply ensure the notebook is in the same folder as the attached datafile and run it.

Note: I will not be attaching a pre-trained model in this folder as there is already one supplied in the Server folder.
