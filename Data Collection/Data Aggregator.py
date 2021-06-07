# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import copy
from datetime import datetime, timedelta

Coins=  ['BTC-USD',    #1
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
coin=Coins[0]

df=pd.DataFrame(columns=['time','low','high','open','close','volume'])
start=datetime(
    year=2020,
    month=1,
    day=1,
    hour=0,
    minute=0,
    second=0)  

end=datetime(
    year=2020,
    month=12,
    day=31,
    hour=0,
    minute=0,
    second=0)

step=timedelta(days=1)

track=copy.deepcopy(start)

while (start <= end):
    filename='Data/Historic '+coin+' data/Historic_'+coin+'_Prices_'+start.strftime('%m_%d_%Y')+'.csv'
    dummy=pd.read_csv(filename,sep=',')
    dummy['date']=start.strftime('%m_%d_%Y')
    df=df.append(dummy,ignore_index=True)
    start=start+step

df.reset_index(inplace=True,drop=True)
df=df.loc[:,~df.columns.str.contains('^Unnamed')]
df.to_csv('Data/Historic '+coin+' data/'+coin+'_Megafile_'
          +track.strftime('%m_%d_%Y')
          +'-'
          +end.strftime('%m_%d_%Y')
          +'.csv')
