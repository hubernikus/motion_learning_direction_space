#!/usr/bin/python3
"""
Directional [SEDS] Learning
"""

__author__ =  "lukashuber"
__date__ = "2021-05-16"

class Learner():
    ''' Virtual class to learn from demonstration / implementation. ''' 
    def __init__(self):
        # Store Data
        self.X = None


    def load_data(self):
        raise NotImplementedError()

    def learn(self):
        raise NotImplementedError()

    def regress(self):
        raise NotImplementedError()

    
