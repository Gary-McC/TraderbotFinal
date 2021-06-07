# -*- coding: utf-8 -*-
import hmac, hashlib, time, base64, csv, copy, sys,time
from datetime import datetime, timedelta
import cbpro
def datemaker(date):#input string, return date from string as list

     
    
    return datetime.strftime(date, '%Y-%m-%dT%H:%M:%S.%fZ') #returns current time from date

def aggregate(coin): #inputs:matrix to hold datapoints,coin name as a string, amount of time to run aggregate over in minutes
    #create controller variables for loop
    start_time=datetime(year=2020,month=1,day=1,hour=0,minute=0,second=0,microsecond=0)
    five_min=timedelta(days=1)
    one_day=timedelta(days=1)
    end_time=datetime(year=2020,month=12,day=31,hour=0,minute=0,second=0,microsecond=0)
#    end_time=start_time+timedelta(days=183)
    
    csvfiller(['Date','Low','High','Open','Close','Volume'],coin, start_time,end_time)
#Create variables that help the loop aggregate
    cur_time=copy.deepcopy(start_time)-one_day
    count=0
    while cur_time < end_time:
        cur_time=cur_time+one_day
        client=cbpro.PublicClient()
        dat=client.get_product_historic_rates(start=datemaker(cur_time),end=datemaker(cur_time+one_day),granularity=86400,product_id=coin)
        track_time=time.localtime(float(dat[0][0]))
        rest=dat[0][1:]
        track=datetime(
                year=track_time.tm_year,
                month=track_time.tm_mon,
                day=track_time.tm_mday,
                hour=0,
                minute=0,
                second=0,
                microsecond=0)
        csvfiller([track.strftime('%m_%d_%Y')]+rest,coin,start_time,end_time)
        track=track-five_min
        count=count+1
        if count==5: #pausing the program every few cycles is for some reason necessary to avoid the program from breaking.
            time.sleep(2)
            count=0
     
                
    return
def csvfiller(Candlestick,Coin,start_time,end_time): #writes data into csv file
    #inputs: data to input, list of coins names   
    with open('Historic_'+Coin+'_Prices_'+start_time.strftime('%m_%d_%Y')+'-'+end_time.strftime('%m_%d_%Y')+'_day_values.csv', 'a+',newline='') as csvfile: 
    # 'a+' allows you to write into an already existing csv
        adder=csv.writer(csvfile,delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
        adder.writerow(Candlestick)
    csvfile.close() 
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


