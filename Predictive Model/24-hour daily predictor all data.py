import hmac, hashlib, time, base64, csv, copy, sys
from datetime import datetime, timedelta
import cbpro,os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

import matplotlib.pyplot as plt
plt.style.use('seaborn')

import warnings
warnings.filterwarnings('ignore')

Coins=  ['BTC-USD',    #1     List of coin names
         'ETH-USD',    #2
         'XRP-USD',    #3
         'LTC-USD',    #4
         'BCH-USD',    #5
         'EOS-USD',    #6
         'DASH-USD',   #7
         'OXT-USD',    #8
         'MKR-USD',    #9
         'XLM-USD',    #10
         'ATOM-USD',   #11
         'XTZ-USD',    #12
         'ETC-USD',    #13
         'OMG-USD',    #14
         'LINK-USD',   #15
         'REP-USD',    #16
         'ZRX-USD',    #17
         'ALGO-USD',   #18
         'DAI-USD',    #19
         'KNC-USD'     #20
         ]
'''---------CONTROL PANEL-----------'''

#HyperParameters
#Name of file to read from
Coin_num=0
Coin=Coins[Coin_num] #Coin file is training for
#Batch Size
batch_size = 32
#Number of ticks network sees at a time
seq_len = 7

d_k = 256
d_v = 256
n_heads = 12
ff_dim = 256
#Modelname= 'Transformer+TimeEmbedding+24hour_offset'+Coin+'-288'
Modelname='Transformer+TimeEmbedding+BTC-USD-24-hour-all'
def datemaker(date):#input string, return date from string as list

     
    s=datetime.strftime(date, '%Y-%m-%dT%H:%M:%S.%fZ')
    return s #returns current time from date

def aggregate(coin): #inputs:matrix to hold datapoints,coin name as a string, amount of time to run aggregate over in minutes
    #create controller variables for loop
   
    five_min=timedelta(minutes=5)
    one_day=timedelta(days=1)
    years=timedelta(days=365)
    end_time=datetime.today()
    start_time=end_time-years
    df=pd.DataFrame(columns=['Time','Low','High','Open','Close','Volume'])
    

#Create variables that help the loop aggregate
    cur_time=copy.deepcopy(start_time)-one_day
    count=0
    while cur_time < end_time:
        cur_time=cur_time+one_day
        client=cbpro.PublicClient()
        dat=client.get_product_historic_rates(start=datemaker(cur_time-one_day),end=datemaker(cur_time),granularity=86400,product_id=coin)
        track_time=time.localtime(float(dat[0][0]))
        track=datetime(
                year=track_time.tm_year,
                month=track_time.tm_mon,
                day=track_time.tm_mday,
                hour=track_time.tm_hour,
                minute=track_time.tm_min,
                second=track_time.tm_sec,
                microsecond=0)
        for i in range(0,len(dat)):
            cut=dat[i]
            raw_time=time.localtime(float(cut[0]))
            rest=cut[1:]
            clean_time=datetime(
                year=raw_time.tm_year,
                month=raw_time.tm_mon,
                day=raw_time.tm_mday,
                hour=raw_time.tm_hour,
                minute=raw_time.tm_min,
                second=raw_time.tm_sec,
                microsecond=0)
            while (track != clean_time):
                df=df.append({'Time':track.strftime('%m_%d_%Y'),'Low':None,'High':None,'Open':None,'Close':None,'Volume':None},ignore_index=True)
                track=track-five_min
            df=df.append({'Time':track.strftime('%m_%d_%Y'),'Low':rest[0],'High':rest[1],'Open':rest[2],'Close':rest[3],'Volume':rest[4]},ignore_index=True)
            track=track-five_min
        count=count+1
        if count==5: #pausing the program every few cycles is for some reason necessary to avoid the program from breaking.
            time.sleep(1)
            count=0
    reverse_df=df.iloc[::-1]
    reverse_df.reset_index(drop=True, inplace=True)
                
    return reverse_df

def csvfiller(Data,Coin): #writes data into csv file
    #inputs: data to input, list of coins names   
    with open('Tomorrow_'+Coin+'_Prices_'+datetime.today().strftime('%m_%d_%Y')+'.csv', 'a+',newline='') as csvfile: 
    # 'a+' allows you to write into an already existing csv
        adder=csv.writer(csvfile,delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
        adder.writerow(Data)
    csvfile.close() 
    return
'''-------------------------------------'''

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

df = aggregate(Coin)
# Replace 0 to avoid dividing by 0 later on
df.columns=['Time','Low','High','Open','Close','Volume']
df['Volume'].replace(to_replace=0, method='ffill', inplace=True) 
df.tail()

# Apply moving average with a window of 15*5 days to all columns
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

###############################################################################
'''Create training, validation and test split'''

times = sorted(df.index.values)
last_10pct = sorted(df.index.values)[-int(0.1*len(times))] # Last 10% of series
last_20pct = sorted(df.index.values)[-int(0.2*len(times))] # Last 20% of series

df_train = df[(df.index < last_20pct)]  # Training data are 80% of total data
df_val = df[(df.index >= last_20pct) & (df.index < last_10pct)]
df_test = df[(df.index >= last_10pct)]

# Remove date column
df_train.drop(columns=['Time'], inplace=True)
df_val.drop(columns=['Time'], inplace=True)
df_test.drop(columns=['Time'], inplace=True)

# Convert pandas columns into arrays
train_data = df_train.values
val_data = df_val.values
test_data = df_test.values
#Moving Average: split data into train, validation, and test set
# Training data
X_train, y_train = [], []
for i in range(seq_len, len(train_data)):
  X_train.append(train_data[i-seq_len:i]) # Chunks of training data with a length of 128 df-rows
  y_train.append(train_data[i][0:4]) #Value of 4th column (Close Price) of df-row 128+1
X_train, y_train = np.array(X_train), np.array(y_train)

###############################################################################

# Validation data
X_val, y_val = [], []
for i in range(seq_len, len(val_data)):
    X_val.append(val_data[i-seq_len:i])
    y_val.append(val_data[i][0:4])
X_val, y_val = np.array(X_val), np.array(y_val)

###############################################################################

# Test data
X_test, y_test = [], []
for i in range(seq_len, len(test_data)):
    X_test.append(test_data[i-seq_len:i])
    y_test.append(test_data[i][0:4])    
X_test, y_test = np.array(X_test), np.array(y_test)

time=datetime.today().strftime('%m_%d_%Y')
model = tf.keras.models.load_model(Modelname+'.hdf5',
                            custom_objects={'Time2Vector':Time2Vector,
                                            'SingleAttention':SingleAttention,
                                            'MultiAttention':MultiAttention,
                                            'TransformerEncoder':TransformerEncoder
                                            })
model.summary()

callback = tf.keras.callbacks.ModelCheckpoint(Modelname+'_'+time+'.hdf5', 
                                              monitor='val_loss', 
                                              save_best_only=True, 
                                              verbose=1)

history = model.fit(X_train, y_train, 
                    batch_size=batch_size, 
                    epochs=40, 
                    callbacks=[callback],
                    validation_data=(X_val, y_val))  

model = tf.keras.models.load_model(Modelname+'_'+time+'.hdf5',
                                   custom_objects={'Time2Vector': Time2Vector, 
                                                   'SingleAttention': SingleAttention,
                                                   'MultiAttention': MultiAttention,
                                                   'TransformerEncoder': TransformerEncoder})


###############################################################################
'''Calculate predictions and metrics'''

#Calculate predication for training, validation and test data
train_pred = model.predict(X_train)
val_pred = model.predict(X_val)
test_pred = model.predict(X_test)
today_pred=test_pred[-1]

#Print evaluation metrics for all datasets
train_eval = model.evaluate(X_train, y_train, verbose=0)
val_eval = model.evaluate(X_val, y_val, verbose=0)
test_eval = model.evaluate(X_test, y_test, verbose=0)
print(' ')
print('Evaluation metrics')
print('Training Data - Loss: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}'.format(train_eval[0], train_eval[1], train_eval[2]))
print('Validation Data - Loss: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}'.format(val_eval[0], val_eval[1], val_eval[2]))
print('Test Data - Loss: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}'.format(test_eval[0], test_eval[1], test_eval[2]))

###############################################################################
'''Display results'''

fig2 = plt.figure(figsize=(15,20))
st = fig2.suptitle("Moving Average - Transformer + TimeEmbedding Model", fontsize=22)
st.set_y(0.92)

#Plot training data results
ax11 = fig2.add_subplot(311)
ax11.plot(train_data[:, 1], label='Coin High')
ax11.plot(np.arange(seq_len, train_pred[:,1].shape[0]+seq_len), train_pred[:,1], linewidth=3, label='Predicted Coin High')
ax11.set_title("Training Data", fontsize=18)
ax11.set_xlabel('Time')
ax11.set_ylabel('Coin High')
ax11.legend(loc="best", fontsize=12)

#Plot validation data results
ax21 = fig2.add_subplot(312)
ax21.plot(val_data[:, 1], label='Coin High')
ax21.plot(np.arange(seq_len, val_pred[:,1].shape[0]+seq_len), val_pred[:,1], linewidth=3, label='Predicted Coin High')
ax21.set_title("Validation Data", fontsize=18)
ax21.set_xlabel('Time')
ax21.set_ylabel('Coin High')
ax21.legend(loc="best", fontsize=12)

#Plot test data results
ax31 = fig2.add_subplot(313)
ax31.plot(test_data[:, 1], label='Coin High')
ax31.plot(np.arange(seq_len, test_pred[:,1].shape[0]+seq_len), test_pred[:,1], linewidth=3, label='Predicted Coin High')
ax31.set_title("Test Data", fontsize=18)
ax31.set_xlabel('Time')
ax31.set_ylabel('Coin High')
ax31.legend(loc="best", fontsize=12)

fig1 = plt.figure(figsize=(15,20))
st = fig1.suptitle("Moving Average - Transformer + TimeEmbedding Model", fontsize=22)
st.set_y(0.92)

#Plot training data results
ax11 = fig1.add_subplot(311)
ax11.plot(train_data[:,0], label='Coin Low')
ax11.plot(np.arange(seq_len, train_pred[:,0].shape[0]+seq_len), train_pred[:,0], linewidth=3, label='Predicted Coin Low')
ax11.set_title("Training Data", fontsize=18)
ax11.set_xlabel('Time')
ax11.set_ylabel('Coin Low')
ax11.legend(loc="best", fontsize=12)

#Plot validation data results
ax21 = fig1.add_subplot(312)
ax21.plot(val_data[:,0], label='Coin Low')
ax21.plot(np.arange(seq_len, val_pred[:,0].shape[0]+seq_len), val_pred[:,0], linewidth=3, label='Predicted Coin Low')
ax21.set_title("Validation Data", fontsize=18)
ax21.set_xlabel('Time')
ax21.set_ylabel('Coin Low')
ax21.legend(loc="best", fontsize=12)

#Plot test data results
ax31 = fig1.add_subplot(313)
ax31.plot(test_data[:,0], label='Coin Low')
ax31.plot(np.arange(seq_len, test_pred[:,0].shape[0]+seq_len), test_pred[:,0], linewidth=3, label='Predicted Coin Low')
ax31.set_title("Test Data", fontsize=18)
ax31.set_xlabel('Time')
ax31.set_ylabel('Coin Low')
ax31.legend(loc="best", fontsize=12)


fig3 = plt.figure(figsize=(15,20))
st = fig3.suptitle("Moving Average - Transformer + TimeEmbedding Model", fontsize=22)
st.set_y(0.92)

#Plot training data results
ax11 = fig3.add_subplot(311)
ax11.plot(train_data[:,2], label='Coin Open')
ax11.plot(np.arange(seq_len, train_pred[:,2].shape[0]+seq_len), train_pred[:,2], linewidth=3, label='Predicted Coin Open')
ax11.set_title("Training Data", fontsize=18)
ax11.set_xlabel('Time')
ax11.set_ylabel('Coin Open')
ax11.legend(loc="best", fontsize=12)

#Plot validation data results
ax21 = fig3.add_subplot(312)
ax21.plot(val_data[:,2], label='Coin Open')
ax21.plot(np.arange(seq_len, val_pred[:,2].shape[0]+seq_len), val_pred[:,2], linewidth=3, label='Predicted Coin Open')
ax21.set_title("Validation Data", fontsize=18)
ax21.set_xlabel('Time')
ax21.set_ylabel('Coin Open')
ax21.legend(loc="best", fontsize=12)

#Plot test data results
ax31 = fig3.add_subplot(313)
ax31.plot(test_data[:,2], label='Coin Open')
ax31.plot(np.arange(seq_len, test_pred[:,2].shape[0]+seq_len), test_pred[:,2], linewidth=3, label='Predicted Coin Open')
ax31.set_title("Test Data", fontsize=18)
ax31.set_xlabel('Time')
ax31.set_ylabel('Coin Open')
ax31.legend(loc="best", fontsize=12)

fig4 = plt.figure(figsize=(15,20))
st = fig4.suptitle("Moving Average - Transformer + TimeEmbedding Model", fontsize=22)
st.set_y(0.92)

#Plot training data results
ax11 = fig4.add_subplot(311)
ax11.plot(train_data[:,3], label='Coin Close')
ax11.plot(np.arange(seq_len, train_pred[:,3].shape[0]+seq_len), train_pred[:,3], linewidth=3, label='Predicted Coin Close')
ax11.set_title("Training Data", fontsize=18)
ax11.set_xlabel('Time')
ax11.set_ylabel('Coin Close')
ax11.legend(loc="best", fontsize=12)

#Plot validation data results
ax21 = fig4.add_subplot(312)
ax21.plot(val_data[:,3], label='Coin Close')
ax21.plot(np.arange(seq_len, val_pred[:,3].shape[0]+seq_len), val_pred[:,3], linewidth=3, label='Predicted Coin Close')
ax21.set_title("Validation Data", fontsize=18)
ax21.set_xlabel('Time')
ax21.set_ylabel('Coin Close')
ax21.legend(loc="best", fontsize=12)

#Plot test data results
ax31 = fig4.add_subplot(313)
ax31.plot(test_data[:,3], label='Coin Close')
ax31.plot(np.arange(seq_len, test_pred[:,3].shape[0]+seq_len), test_pred[:,3], linewidth=3, label='Predicted Coin Close')
ax31.set_title("Test Data", fontsize=18)
ax31.set_xlabel('Time')
ax31.set_ylabel('Coin Close')
ax31.legend(loc="best", fontsize=12)

today=[datetime.today().strftime('%m_%d_%Y')]
for i in today_pred:
    if i<= 0.5:
        today.append(-1)
    else:
        today.append(1)
csvfiller(['Date','Low','High','Open','Close'],Coin)
csvfiller(today, Coin)
tomorrow_pred=model.predict(test_data[-seq_len:])
tomorrow=[(datetime.today()+timedelta(days=1)).strftime('%m_%d_%Y')]
for i in tomorrow_pred:
    if i<= 0.5:
        tomorrow.append(-1)
    else:
        tomorrow.append(1)
csvfiller(tomorrow, Coin)
