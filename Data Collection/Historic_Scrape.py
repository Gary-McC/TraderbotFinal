# -*- coding: utf-8 -*-
import hmac, hashlib, time, base64, csv, copy, sys, time
from datetime import datetime, timedelta
import cbpro
import pandas as pd
def datemaker(date):#input string, return date from string as list

     
    s=datetime.strftime(date, '%Y-%m-%dT%H:%M:%S.%fZ')
    return s #returns current time from date

def datefixer(date):#input date, return cleaned time for final dataframe
    track_time=time.localtime(float(date))
    track=datetime(
                year=track_time.tm_year,
                month=track_time.tm_mon,
                day=track_time.tm_mday,
                hour=track_time.tm_hour,
                minute=track_time.tm_min,
                second=track_time.tm_sec,
                microsecond=0)

    return track.strftime('%H:%M:%S')

def aggregate(coin): #inputs:matrix to hold datapoints,coin name as a string, amount of time to run aggregate over in minutes
    #create controller variables for loop
    start_time=datetime(year=2020,month=1,day=1,hour=0,minute=0,second=0,microsecond=0)
    five_min=timedelta(minutes=5)
    one_day=timedelta(days=1)
    end_time=datetime(year=2021,month=1,day=1,hour=0,minute=0,second=0,microsecond=0)

#Create variables that help the loop aggregate
    cur_time=copy.deepcopy(start_time)-one_day
    while cur_time < end_time:
        cur_time=cur_time+one_day
        print((cur_time < end_time), '\nCurrent time: ', cur_time, '   End time: ',end_time)
        client=cbpro.PublicClient()
        time.sleep(1)
        dat=client.get_product_historic_rates(start=datemaker(cur_time),end=datemaker(cur_time+one_day),granularity=300,product_id=coin)
        df=pd.DataFrame(dat,columns=['Time','Low','High','Open','Close','Volume'])
        df.Time=df.Time.map(datefixer) #fix date format
        df=df.iloc[::-1] #invert dataframe 
        track_time=time.localtime(float(dat[0][0]))
        track=datetime(
                year=track_time.tm_year,
                month=track_time.tm_mon,
                day=track_time.tm_mday,
                hour=track_time.tm_hour,
                minute=track_time.tm_min,
                second=track_time.tm_sec,
                microsecond=0)
        df.to_csv('Historic_'+coin+'_Prices_'+track.strftime('%m_%d_%Y')+'.csv')
        
     
                
    return


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
aggregate(Coins[0])


