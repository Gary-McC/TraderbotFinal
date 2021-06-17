# Data Collection

For the purposes of this project, it is necessary to gather data in both 5-minute and 24-hour ticks. 5-minute ticks to train the RL Agent, and 24-hour ticks for the predictive model. While it is certainly possible to train a predictive model on the five minute ticks, and can be done with trivial modifications to the predictive model generating code in Predictive Models, the bot itself uses a 24-hour prediction both to reduce the computational load on the machine, and to maximize the impact of the prediction itself; looking ahead 24-hours is far more useful than looking ahead five minutes. 

In this folder is the code necessary to gather, clean, and aggregate this data for use. All data gathered is gathered from the pro.coinbase.com using the cbpro API.

# [5 Minute Historic Scrape.py](https://github.com/Gary-McC/TraderbotFinal/blob/main/Data%20Collection/24-hour%20Datamaker.py)

This file reads and saves historical cryptocurrency price data into csv files. Each csv file contains one day's worth of data. I found this to be more useful than saving multiple days worth of data into one csv file as I could easily aggregate the csv data into larger files once it was collected. This is much faster than collecting potentially the same data multiple times for multiple different timespans, and allows the user to work offline once the data is collected. To run, simply adjust the start and end time variables to the time window of your choice, then select the coin you would like to read from.

# [Data Aggregator.py](https://github.com/Gary-McC/TraderbotFinal/blob/main/Data%20Collection/Data%20Aggregator.py)

This file aggregates data from multiple daily 5-minute csv files into one large megafile. Running this is fairly straightforward, simply adjust the start and end time ranges to reflect the time range your want you large datafile to reflect. It is important to ensure you've already collected data over the timeframe you have chosen to cover when running this file, otherwise it will fail.

# [24-hour datamaker.py](https://github.com/Gary-McC/TraderbotFinal/blob/main/Data%20Collection/24-hour%20Datamaker.py)

This program collects data in 24-hour ticks and saves it to a csv file. Because we are already collecting daily tick data, it is much less inefficient to directly save all of the data into a single csv file rather than saving it into a separate csv file for each individual day.Unlike with collecting 5 minute data, it is much easier to collect the data one tick at a time to avoid crashing the connection with the API. I am unsure why this is, but this method of gathering daily tick data, while notably slower, is much more consistently stable. To run, simply adjust the start and end dates of the days you wish to cover. 

# [Historic Data](https://github.com/Gary-McC/TraderbotFinal/tree/main/Data%20Collection/Historic%20Data)

For your convenience, I am providing some previously collected BTC-USD data, including some aggregated 5-minute data(ie:megafiles).
