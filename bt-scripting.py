from __future__ import (absolute_import, division, print_function, unicode_literals)
import datetime
import backtrader as bt
#from gprStrategy import gprStrategy
from introStrategy import introStrategy

import backtrader as bt
from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import WhiteKernel

class gprStrategy(bt.Strategy):

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close
        self.dataopen = self.datas[0].open

        # To keep track of pending orders
        self.order = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('BUY EXECUTED, %.2f' % order.executed.price)
            elif order.issell():
                self.log('SELL EXECUTED, %.2f' % order.executed.price)

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Write down: no pending order
        self.order = None

    def next(self):
        window_size = 100
        kernel = 1.0**2 * RBF(length_scale=24.0) + 10.0**2 * WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-20,1e-1))
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=5, normalize_y=True)

        self.log('Close, %.2f' % self.dataclose[0])

        open_price = self.dataopen[0]

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return
        
        X = np.arange(window_size)[:, None]
        y = np.array([self.dataclose[-i] for i in reversed(range(1, window_size + 1))]).reshape(-1,1)

        gpr.fit(X, y)
        y_pred, std = gpr.predict([X[-1]], return_std=True)

        # Check if we are in the market
        if not self.position:
            # Not yet ... we MIGHT BUY if ...
            if open_price <= (y_pred - 1.96 * std):
                self.log('BUY CREATE, %.2f' % self.dataclose[0])
                self.order = self.buy()

            elif open_price >= (y_pred + 1.96 * std):
                self.log('SELL CREATE, %.2f' % self.dataclose[0])
                self.order = self.sell()

        elif self.position.size > 0:
            if open_price >= y_pred:
                self.log('CLOSE CREATE, %.2f' % self.dataclose[0])
                self.order = self.close()

        elif self.position.size < 0:
            if open_price <= y_pred:
                self.log('CLOSE CREATE, %.2f' % self.dataclose[0])
                self.order = self.close()

cerebro = bt.Cerebro()

cerebro.addstrategy(gprStrategy)

data = bt.feeds.GenericCSVData(
    dataname='GBPUSD_H1.csv',

    fromdate=datetime.datetime(2018, 3, 1),
    todate=datetime.datetime(2018, 3, 30),

    nullvalue=0.0,

    dtformat=('%Y-%m-%d %H:%M'),

    datetime=0,
    high=2,
    low=3,
    open=1,
    close=4,
    volume=5,
    openinterest=-1
)

cerebro.adddata(data)
cerebro.broker.setcash(1000.0)

print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
cerebro.run()
print('Ending Portfolio Value: %.2f' % cerebro.broker.getvalue())

cerebro.plot()