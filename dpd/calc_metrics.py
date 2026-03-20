# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:17:40 2026

@author: PC
"""

import numpy as np

def calcPAPR(signal):
    peak_power = np.max(np.abs(signal)) ** 2
    average_power = np.mean(np.abs(signal) ** 2)
    
    papr = peak_power / average_power
    
    return 10 * np.log10(papr)


def calcACLR(Psd, freqs, B, offset):
    df = freqs[1] - freqs[0]
    
    Pin = np.sum(Psd[freqs >= - B] * df) - np.sum(Psd[freqs >= B] * df)
    
    Pout1 = np.sum(Psd[ freqs <= -B - offset] * df) - np.sum(Psd[ freqs <= -3*B - offset] * df)
    Pout2 = np.sum(Psd[ freqs >= B + offset] * df) - np.sum(Psd[ freqs >= 3*B + offset] * df)
    
    Pout = np.max([Pout1, Pout2])
    
    return 10*np.log10(Pout / Pin)


def calcMSE(x, y):    
    return np.mean(np.abs(x - y)**2)