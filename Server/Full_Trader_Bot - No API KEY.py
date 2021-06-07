from gym.spaces import Discrete
import ta
from tensortrade.env.default.actions import TensorTradeActionScheme
import ray.rllib.agents.ppo as ppo
from tensortrade.env.generic import ActionScheme, TradingEnv
from tensortrade.core import Clock
from tensortrade.oms.instruments import ExchangePair
from tensortrade.oms.wallets import Portfolio
from tensortrade.oms.orders import (
    Order,
    proportion_order,
    TradeSide,
    TradeType
)
import csv
import ray
import pandas as pd
import numpy as np
from ray import tune
from ray.tune.registry import register_env
import tensortrade.env.default as default
from tensortrade.feed.core import DataFeed, Stream
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Wallet, Portfolio
import hmac, hashlib, time, base64, csv, copy, sys,time
from datetime import datetime, timedelta
import cbpro
from tensortrade.env.default.rewards import TensorTradeRewardScheme
from tensortrade.feed.core import Stream, DataFeed
from tensortrade.oms.instruments import Instrument
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import matplotlib.pyplot as plt
from tensortrade.env.generic import Renderer


class PositionChangeChart(Renderer):

    def __init__(self, color: str = "orange"):
        self.color = "orange"

    def render(self, env, **kwargs):
        history = pd.DataFrame(env.observer.renderer_history)

        actions = list(history.action)
        p = list(history.price)

        buy = {}
        sell = {}

        for i in range(len(actions) - 1):
            a1 = actions[i]
            a2 = actions[i + 1]

            if a1 != a2:
                if a1 == 0 and a2 == 1:
                    buy[i] = p[i]
                else:
                    sell[i] = p[i]

        buy = pd.Series(buy)
        sell = pd.Series(sell)

        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        fig.suptitle("Performance")

        axs[0].plot(np.arange(len(p)), p, label="price", color=self.color)
        axs[0].scatter(buy.index, buy.values, marker="^", color="green")
        axs[0].scatter(sell.index, sell.values, marker="^", color="red")
        axs[0].set_title("Trading Chart")
        
        performance = pd.DataFrame.from_dict(env.action_scheme.portfolio.performance, orient='index')
        performance.plot(ax=axs[1])
        axs[1].set_title("Net Worth")

        plt.show()
class BSH(TensorTradeActionScheme):

    registered_name = "bsh"

    def __init__(self, cash: 'Wallet', asset: 'Wallet'):
        
        super().__init__()
        self.cash = cash
        self.asset = asset

        self.listeners = []
        self.action = 0

    @property
    def action_space(self):
        return Discrete(2)

    def attach(self, listener):
        self.listeners += [listener]
        return self

    def get_orders(self, action: int, portfolio: 'Portfolio'):
        order = None

        if abs(action - self.action) > 0:
            src = self.cash if self.action == 0 else self.asset
            tgt = self.asset if self.action == 0 else self.cash
            order = proportion_order(portfolio, src, tgt, 1.0)
            self.action = action

        for listener in self.listeners:
            listener.on_action(action)

        return [order]

    def reset(self):
        super().reset()
        self.action = 0
#Custom classes for transformer predictor:
#Timevector layer
class Time2Vector(Layer):
  def __init__(self, seq_len, **kwargs):
    super(Time2Vector, self).__init__()
    self.seq_len = seq_len

  def build(self, input_shape):
    '''Initialize weights and biases with shape (batch, seq_len)'''
    self.weights_linear = self.add_weight(name='weight_linear',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)
    
    self.bias_linear = self.add_weight(name='bias_linear',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)
    
    self.weights_periodic = self.add_weight(name='weight_periodic',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)

    self.bias_periodic = self.add_weight(name='bias_periodic',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)

  def call(self, x):
    '''Calculate linear and periodic time features'''
    x = tf.math.reduce_mean(x[:,:,:4], axis=-1) 
    time_linear = self.weights_linear * x + self.bias_linear # Linear time feature
    time_linear = tf.expand_dims(time_linear, axis=-1) # Add dimension (batch, seq_len, 1)
    
    time_periodic = tf.math.sin(tf.multiply(x, self.weights_periodic) + self.bias_periodic)
    time_periodic = tf.expand_dims(time_periodic, axis=-1) # Add dimension (batch, seq_len, 1)
    return tf.concat([time_linear, time_periodic], axis=-1) # shape = (batch, seq_len, 2)
   
  def get_config(self): # Needed for saving and loading model with custom layer
    config = super().get_config().copy()
    config.update({'seq_len': self.seq_len})
    return config

#Transformer Layers
class SingleAttention(Layer):
  def __init__(self, d_k, d_v):
    super(SingleAttention, self).__init__()
    self.d_k = d_k
    self.d_v = d_v

  def build(self, input_shape):
    self.query = Dense(self.d_k, 
                       input_shape=input_shape, 
                       kernel_initializer='glorot_uniform', 
                       bias_initializer='glorot_uniform')
    
    self.key = Dense(self.d_k, 
                     input_shape=input_shape, 
                     kernel_initializer='glorot_uniform', 
                     bias_initializer='glorot_uniform')
    
    self.value = Dense(self.d_v, 
                       input_shape=input_shape, 
                       kernel_initializer='glorot_uniform', 
                       bias_initializer='glorot_uniform')

  def call(self, inputs): # inputs = (in_seq, in_seq, in_seq)
    q = self.query(inputs[0])
    k = self.key(inputs[1])

    attn_weights = tf.matmul(q, k, transpose_b=True)
    attn_weights = tf.map_fn(lambda x: x/np.sqrt(self.d_k), attn_weights)
    attn_weights = tf.nn.softmax(attn_weights, axis=-1)
    
    v = self.value(inputs[2])
    attn_out = tf.matmul(attn_weights, v)
    return attn_out    

#############################################################################

class MultiAttention(Layer):
  def __init__(self, d_k, d_v, n_heads):
    super(MultiAttention, self).__init__()
    self.d_k = d_k
    self.d_v = d_v
    self.n_heads = n_heads
    self.attn_heads = list()

  def build(self, input_shape):
    for n in range(self.n_heads):
      self.attn_heads.append(SingleAttention(self.d_k, self.d_v))  
    
    # input_shape[0]=(batch, seq_len, 7), input_shape[0][-1]=7 
    self.linear = Dense(input_shape[0][-1], 
                        input_shape=input_shape, 
                        kernel_initializer='glorot_uniform', 
                        bias_initializer='glorot_uniform')

  def call(self, inputs):
    attn = [self.attn_heads[i](inputs) for i in range(self.n_heads)]
    concat_attn = tf.concat(attn, axis=-1)
    multi_linear = self.linear(concat_attn)
    return multi_linear   

#############################################################################

class TransformerEncoder(Layer):
  def __init__(self, d_k, d_v, n_heads, ff_dim, dropout=0.1, **kwargs):
    super(TransformerEncoder, self).__init__()
    self.d_k = d_k
    self.d_v = d_v
    self.n_heads = n_heads
    self.ff_dim = ff_dim
    self.attn_heads = list()
    self.dropout_rate = dropout

  def build(self, input_shape):
    self.attn_multi = MultiAttention(self.d_k, self.d_v, self.n_heads)
    self.attn_dropout = Dropout(self.dropout_rate)
    self.attn_normalize = LayerNormalization(input_shape=input_shape, epsilon=1e-6)

    self.ff_conv1D_1 = Conv1D(filters=self.ff_dim, kernel_size=1, activation='relu')
    # input_shape[0]=(batch, seq_len, 7), input_shape[0][-1] = 7 
    self.ff_conv1D_2 = Conv1D(filters=input_shape[0][-1], kernel_size=1) 
    self.ff_dropout = Dropout(self.dropout_rate)
    self.ff_normalize = LayerNormalization(input_shape=input_shape, epsilon=1e-6)    
  
  def call(self, inputs): # inputs = (in_seq, in_seq, in_seq)
    attn_layer = self.attn_multi(inputs)
    attn_layer = self.attn_dropout(attn_layer)
    attn_layer = self.attn_normalize(inputs[0] + attn_layer)

    ff_layer = self.ff_conv1D_1(attn_layer)
    ff_layer = self.ff_conv1D_2(ff_layer)
    ff_layer = self.ff_dropout(ff_layer)
    ff_layer = self.ff_normalize(inputs[0] + ff_layer)
    return ff_layer 

  def get_config(self): # Needed for saving and loading model with custom layer
    config = super().get_config().copy()
    config.update({'d_k': self.d_k,
                   'd_v': self.d_v,
                   'n_heads': self.n_heads,
                   'ff_dim': self.ff_dim,
                   'attn_heads': self.attn_heads,
                   'dropout_rate': self.dropout_rate})
    return config
USD = Instrument("USD", 2, "U.S. Dollar")
TTC = Instrument("TTC", 8, "TensorTrade Coin")

class MXRewardScheme(TensorTradeRewardScheme):
    """A reward scheme that rewards the agent for increasing its net worth,
        while penalizing more volatile strategies.
        Parameters
        ----------
        :param: return_algorithm : {'sharpe', 'sortino'}, Default 'sharpe'.
            The risk-adjusted return metric to use.
        :param: risk_free_rate : float, Default 0.
            The risk free rate of returns to use for calculating metrics.
        :param: target_returns : float, Default 0
            The target returns per period for use in calculating the sortino ratio.
        :param: window_size : int
            The size of the look back window for computing the reward.
        """

    def __init__(self,
                 return_algorithm: str = 'sharpe',
                 risk_free_rate: float = 0.,
                 target_returns: float = 0.,
                 window_size: int = 1) -> None:
        algorithm = self.default('return_algorithm', return_algorithm)

        assert algorithm in ['sharpe', 'diff_sharpe', 'sortino']

        if algorithm == 'sharpe':
            return_algorithm = self._sharpe_ratio
        elif algorithm == 'sortino':
            return_algorithm = self._sortino_ratio
        elif algorithm == 'diff_sharpe':
            return_algorithm = self._diff_sharpe_ratio

        self._return_algorithm = return_algorithm
        self._risk_free_rate = self.default('risk_free_rate', risk_free_rate)
        self._target_returns = self.default('target_returns', target_returns)
        self._window_size = self.default('window_size', window_size)

    def _sharpe_ratio(self, returns: 'pd.Series') -> float:
        """Computes the sharpe ratio for a given series of a returns.
        Parameters
        ----------
        returns : `pd.Series`
            The returns for the `portfolio`.
        Returns
        -------
        float
            The sharpe ratio for the given series of a `returns`.
        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Sharpe_ratio
        """
        return (np.mean(returns) - self._risk_free_rate + 1e-9) / (np.std(returns) + 1e-9)

    def _diff_sharpe_ratio(self, returns: 'pd.Series') -> float:
        """Computes the differential sharpe ratio over a given series of returns
        Parameters
        ----------
        returns : `pd.Series`
            The returns for the 'portfolio'
        Returns
        -------
        float
            The differential sharpe ratio for the given series of a `returns`
        References
        ----------
        .. [1] https://proceedings.neurips.cc/paper/1998/file/4e6cd95227cb0c280e99a195be5f6615-Paper.pdf
        .. [2] https://github.com/AchillesJJ/DSR
        """
        np.seterr('raise')
        eta = 0.004

        A = np.mean(returns[-1:])
        B = np.mean(returns[-1:]**2)
        delta_A = np.mean(returns) - A
        delta_B = np.mean(returns)**2 - B
        upper = ((B * delta_A - 0.5*A*delta_B) + 1e-9)
        lower = (B-A**2 + 1e-9)**(3/2)

        if lower == (0 or np.isnan(lower)):
            print(f"A:{A}\n"
                  f"B:{B}\n"
                  f"delta_A:{delta_A}\n"
                  f"delta_B:{delta_B}\n"
                  f"upper:{upper}\n"
                  f"lower:{lower}\n")
                  # f"reward:{dt*eta}\n")
        dt = upper / lower

        return dt * eta

    def _sortino_ratio(self, returns: 'pd.Series') -> float:
        """Computes the sortino ratio for a given series of a returns.
        Parameters
        ----------
        returns : `pd.Series`
            The returns for the `portfolio`.
        Returns
        -------
        float
            The sortino ratio for the given series of a `returns`.
        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Sortino_ratio
        """
        downside_returns = returns.copy()
        downside_returns[returns < self._target_returns] = returns ** 2

        expected_return = np.mean(returns)
        downside_std = np.sqrt(np.std(downside_returns))

        result = (expected_return - self._risk_free_rate + 1e-9) / (downside_std + 1e-9)

        return result

    def get_reward(self, portfolio: 'Portfolio') -> float:
        """Computes the reward corresponding to the selected risk-adjusted return metric.
        Parameters
        ----------
        portfolio : `Portfolio`
            The current portfolio being used by the environment.
        Returns
        -------
        float
            The reward corresponding to the selected risk-adjusted return metric.
        """
        net_worths = [nw['net_worth'] for nw in portfolio.performance.values()][-(self._window_size + 1):]
        returns = pd.Series(net_worths).pct_change().dropna()
        risk_adjusted_return = self._return_algorithm(returns)
        return risk_adjusted_return

class PBR(TensorTradeRewardScheme):

    registered_name = "pbr"

    def __init__(self, price: 'Stream'):
        super().__init__()
        self.position = -1

        r = Stream.sensor(price, lambda p: p.value, dtype="float").diff()
        position = Stream.sensor(self, lambda rs: rs.position, dtype="float")

        reward = (r * position).fillna(0).rename("reward")

        self.feed = DataFeed([reward])
        self.feed.compile()

    def on_action(self, action: int):
        self.position = -1 if action == 0 else 1

    def get_reward(self, portfolio: 'Portfolio'):
        return self.feed.next()["reward"]

    def reset(self):
        self.position = -1
        self.feed.reset()
start_position = pd.read_csv('Starting Wallet.csv')
wallet=[start_position.USD[0],start_position.BTC[0]]
#generate data
def datemaker(date):#input string, return date from string as list
    return datetime.strftime(date, '%Y-%m-%dT%H:%M:%S.%fZ') #returns current time from date

cur_time=datetime.today()
one_day=timedelta(days=1)
five_minutes=timedelta(minutes=5*(int(start_position['time offset'][0]-1)))
client=cbpro.PublicClient()
dat=pd.DataFrame(client.get_product_historic_rates(start=datemaker(cur_time-one_day-timedelta(days=4)+five_minutes),end=datemaker(cur_time-timedelta(days=4)),granularity=300,product_id='BTC-USD')
                ,columns=['time','low','high','open','close','volume'])
dataframe=pd.DataFrame(columns=['time','low','high','open','close','volume'])
dataframe=dataframe.append(dat.iloc[::-1])
for i in range(1,5):
    dataframe=dataframe.append(pd.DataFrame(
    client.get_product_historic_rates(start=datemaker(cur_time-one_day-timedelta(days=4-i)),end=datemaker(cur_time-timedelta(days=4-i)),granularity=300,product_id='BTC-USD')
                ,columns=['time','low','high','open','close','volume']).iloc[::-1])
    dataframe.reset_index(drop=True, inplace=True)

def replace(x):
    t=time.localtime(x)
    return datetime(year=t.tm_year,month=t.tm_mon,day=t.tm_mday).strftime('%d-%m-%Y')
#dataframe.time=dataframe.time.map(replace)
#flip dataframe so first value is the oldest
dataframe.time=dataframe.time.map(replace)
dataframe.head()

#predictor:
#need more than one day for proper averaging of columns
df=pd.DataFrame(client.get_product_historic_rates(start=datemaker(cur_time-timedelta(days=25)),end=datemaker(cur_time),granularity=86400,product_id='BTC-USD')
                  ,columns=['Time','Low','High','Open','Close','Volume'])
# Apply moving average with a window of 15*5 days to all columns
df=df.iloc[::-1]
df[['Low','High','Open','Close','Volume']] = df[['Low','High','Open','Close','Volume']].rolling(15).mean() 
# Drop all rows with NaN values
df.dropna(how='any', axis=0, inplace=True) 
#Moving Average, calculate percentage change for columns
df['Open'] = df['Open'].pct_change() # Create arithmetic returns column
df['High'] = df['High'].pct_change() # Create arithmetic returns column
df['Low'] = df['Low'].pct_change() # Create arithmetic returns column
df['Close'] = df['Close'].pct_change() # Create arithmetic returns column
df['Volume'] = df['Volume'].pct_change()

df.dropna(how='any', axis=0, inplace=True) # Drop all rows with NaN values

###############################################################################
'''Normalize price columns'''

min_return = min(df[['Open', 'High', 'Low', 'Close']].min(axis=0))
max_return = max(df[['Open', 'High', 'Low', 'Close']].max(axis=0))

# Min-max normalize price columns (0-1 range)
df['Open'] = (df['Open'] - min_return) / (max_return - min_return)
df['High'] = (df['High'] - min_return) / (max_return - min_return)
df['Low'] = (df['Low'] - min_return) / (max_return - min_return)
df['Close'] = (df['Close'] - min_return) / (max_return - min_return)

###############################################################################
'''Normalize volume column'''

min_volume = df['Volume'].min(axis=0)
max_volume = df['Volume'].max(axis=0)

# Min-max normalize volume columns (0-1 range)
df['Volume'] = (df['Volume'] - min_volume) / (max_volume - min_volume)
df.Time=df.Time.map(replace)
df=df.iloc[3:]
#df
#remove time column:
times=df['Time']
df.drop(columns=['Time'],inplace=True)
#load and generate prediction: 
Modelname='Transformer+TimeEmbedding+BTC-USD-24-hour-all'
model = tf.keras.models.load_model(Modelname+'.hdf5',
                                   custom_objects={'Time2Vector': Time2Vector, 
                                                   'SingleAttention': SingleAttention,
                                                   'MultiAttention': MultiAttention,
                                                   'TransformerEncoder': TransformerEncoder})
df.head()
times=times.append(pd.Series(datetime.today().strftime('%d_%m_%Y')))
test=(model.predict(np.array([df])))
df=df.append(pd.DataFrame(test, columns=['Close','High','Low','Open']))
for row in df:
    pred=[]
    for item in df[row]:
        if item <= 0.5:
            pred.append(-1)
        else:
            pred.append(1)
    df[row]=pd.Series(pred)
df['time']=times
df.reset_index(inplace=True)
df.drop(columns=['Volume','index'],inplace=True)
df.columns=['rel_low','rel_high','rel_open','rel_close','time']
data=pd.merge(dataframe,df,on='time',how='inner')




def create_env(config):
    data=config["data"]
    c = Stream.source(list(data["close"]), dtype="float").rename("USD-TTC")
    #sharpe=Stream.source(list(c_sharpe_ratio), dtype="float").rename("sharpe_ratio")
    #vola=Stream.source(list(c_volatility), dtype="float").rename("volatility")
    
    o = Stream.source(list(data["open"]), dtype="float").rename("open")
    h = Stream.source(list(data["high"]), dtype="float").rename("high")
    l = Stream.source(list(data["low"]), dtype="float").rename("low")
    v = Stream.source(list(data["volume"]), dtype="float").rename("volume")
    bitfinex = Exchange("bitfinex", service=execute_order)(
        c
    )

    cash = Wallet(bitfinex, config["USD"] * USD)
    asset = Wallet(bitfinex, config["BTC"] * TTC)

    portfolio = Portfolio(USD, [
        cash,
        asset
    ])

    feed = DataFeed([
    o,
    #o.rolling(window=10).mean().rename("o_fast"),
    #o.rolling(window=50).mean().rename("o_medium"),
    #o.rolling(window=100).mean().rename("o_slow"),
    o.log().diff().fillna(0).rename("o_lr"),
        
    h,
    #h.rolling(window=10).mean().rename("h_fast"),
    #h.rolling(window=50).mean().rename("h_medium"),
    #h.rolling(window=100).mean().rename("h_slow"),
    h.log().diff().fillna(0).rename("h_lr"),
    
        
    l,
    #l.rolling(window=10).mean().rename("l_fast"),
    #l.rolling(window=50).mean().rename("l_medium"),
    #l.rolling(window=100).mean().rename("l_slow"),
    l.log().diff().fillna(0).rename("l_lr"),
        
    c,
    #c.rolling(window=10).mean().rename("c_fast"),
    #c.rolling(window=50).mean().rename("c_medium"),
    #c.rolling(window=100).mean().rename("c_slow"),
    c.log().diff().fillna(0).rename("c_lr"), 
    
    v, 
    Stream.source(list(data["rel_low"]), dtype="float").rename("rel_low"),
    Stream.source(list(data["rel_high"]), dtype="float").rename("rel_high"),
    Stream.source(list(data["rel_open"]), dtype="float").rename("rel_open"),
    Stream.source(list(data["rel_close"]), dtype="float").rename("rel_close"),
    ])

    #reward_scheme = MXRewardScheme(return_algorithm='diff_sharpe')
    reward_scheme = PBR(price=o)

    action_scheme = BSH(
        cash=cash,
        asset=asset
    ).attach(reward_scheme) #remove the .attach for non PBR reward schemes

    renderer_feed = DataFeed([
        Stream.source(list(data["close"]), dtype="float").rename("price"),
        Stream.sensor(action_scheme, lambda s: s.action, dtype="float").rename("action")
    ])

    environment = default.create(
        feed=feed,
        portfolio=portfolio,
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        renderer_feed=renderer_feed,
        renderer=PositionChangeChart(),
        window_size=config["window_size"],
        max_allowed_loss=0.6
    )
    return environment

register_env("TradingEnv", create_env)

checkpoint_path = 'RL Agent Model/checkpoint_000004/checkpoint-4'

# Restore agent
ray.init()
agent = ppo.PPOTrainer(
    env="TradingEnv",
    config={
        "env_config": {
            "window_size": 25,
            "data":data,
            "USD":wallet[0],
            "BTC":wallet[1]
        },
        "framework": "torch",
        "log_level": "DEBUG",
        "ignore_worker_failures": True,
        "num_workers": 1,
        "num_gpus": 0,
        "clip_rewards": True,
        "lr": 8e-7,
        "lr_schedule": [
            [0, 1e-1],
            [int(1e2), 1e-2],
            [int(1e3), 1e-3],
            [int(1e4), 1e-4],
            [int(1e5), 1e-5],
            [int(1e6), 1e-6],
            [int(1e7), 1e-7]
        ],
        "gamma": 0,
        "observation_filter": "MeanStdFilter",
        "lambda": 0.72,
        "vf_loss_coeff": 0.5,
        "entropy_coeff": 0.01
    }
)
agent.restore(checkpoint_path)

# Instantiate the environment
env = create_env({
    "window_size": 25,
    "data":data,
    "USD":wallet[0],
    "BTC":wallet[1]
})

# Run until episode ends
episode_reward = 0
done = False
obs = env.reset()

while not done:
    action = agent.compute_action(obs)
    obs, reward, done, info = env.step(action)
    episode_reward += reward
history = pd.DataFrame(env.observer.renderer_history)
actions=list(history.action)


#Save wallet state at time point equals 2
performance = pd.DataFrame.from_dict(env.action_scheme.portfolio.performance, orient='index')
for i in range(1,len(performance)):
    if performance['bitfinex:/USD:/total'][i]!=0:
        balance=[performance['bitfinex:/USD:/total'][i],performance['bitfinex:/TTC:/total'][i],i,]
        break
with open('Starting Wallet.csv', 'w',newline='') as csvfile: 
        adder=csv.writer(csvfile,delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
        adder.writerow(['USD','BTC','time offset'])
        adder.writerow(balance)
csvfile.close() 
ray.shutdown()
#Buy and Sell Logic here

action=actions[-2:]
if action[0] != action[1]:
    client=cbpro.AuthenticatedClient('API KEY'
                                     , 'API SECRET'
                                     , 'API PASS')
    wallet=pd.DataFrame.from_dict(client.get_accounts())
    if action[0]==1:
        client.sell(
                size=wallet[wallet.currency=='BTC'].balance,
                order_type='market',
                product_id='BTC-USD'
                )
        with open('Bot Logs.csv', '+a', newline='') as csvfile:
            adder=csv.writer(csvfile,delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
            adder.writerow(['Sell',datetime.today().strftime('%h-%d-%Y %H:%M'),performance['bitfinex:/USD:/total'].iloc[-1],performance['bitfinex:/TTC:/total'].iloc[-1]])
        csvfile.close() 
    else:
        client.buy(
                funds=wallet[wallet.currency=='USD'].balance,
                order_type='market',
                product_id='BTC-USD'
                )
        with open('Bot Logs.csv', '+a', newline='') as csvfile:
            adder=csv.writer(csvfile,delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
            adder.writerow(['Buy',datetime.today().strftime('%h-%d-%Y %H:%M'),performance['bitfinex:/USD:/total'].iloc[-1],performance['bitfinex:/TTC:/total'].iloc[-1]])
        csvfile.close()
else:
    with open('Bot Logs.csv', '+a', newline='') as csvfile:
        adder=csv.writer(csvfile,delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
        adder.writerow(['None',datetime.today().strftime('%h-%d-%Y %H:%M'),performance['bitfinex:/USD:/total'].iloc[-1],performance['bitfinex:/TTC:/total'].iloc[-1]])
    csvfile.close()