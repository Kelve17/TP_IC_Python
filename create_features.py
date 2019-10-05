# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 10:27:47 2019

@author: Kelve Neto
"""

import numpy as np
import math as mt
from statsmodels import robust
from scipy.stats import skew, kurtosis
from statsmodels.tsa import stattools
from statistics import median
from statistics import mode
from scipy.stats import iqr
# Import PySwarms
from iteration_utilities import deepflatten
import csv


def extract_features():
    dados = []
    for x in range(1,23):
            fich = np.loadtxt("%d.csv" %x,delimiter=",");
            aux = []
            auxAcX = []
            auxAcY = []
            auxAcZ = []
            valor = 2;
            for y in range(0,len(fich)):
                if fich[y][0] <= valor and y != len(fich)-1:
                    auxAcX.append(fich[y,1])
                    auxAcY.append(fich[y,2])
                    auxAcZ.append(fich[y,3])
                elif fich[y,0] > valor or y == len(fich)-1:
                    try:
                         aux = [[x-1,np.mean(auxAcX),np.std(auxAcX),np.var(auxAcX),median(auxAcX),np.percentile(auxAcX,25),np.percentile(auxAcX,75),mode(auxAcX),np.min(auxAcX),np.argmin(auxAcX),np.max(auxAcX),np.argmax(auxAcX),robust.mad(auxAcX),stattools.acf(auxAcX).mean(),stattools.acf(auxAcX).std(),stattools.acovf(auxAcX).mean(),stattools.acovf(auxAcX).std(),skew(auxAcX),kurtosis(auxAcX),iqr(auxAcX),
                                     np.mean(auxAcY),np.std(auxAcY),np.var(auxAcY),median(auxAcY),np.percentile(auxAcY,25),np.percentile(auxAcY,75),mode(auxAcY),np.min(auxAcY),np.argmin(auxAcY),np.max(auxAcY),np.argmax(auxAcY),robust.mad(auxAcY),stattools.acf(auxAcY).mean(),stattools.acf(auxAcX).std(),stattools.acovf(auxAcY).mean(),stattools.acovf(auxAcY).std(),skew(auxAcY),kurtosis(auxAcY),iqr(auxAcY),
                                     np.mean(auxAcZ),np.std(auxAcZ),np.var(auxAcZ),median(auxAcZ),np.percentile(auxAcZ,25),np.percentile(auxAcZ,75),mode(auxAcZ),np.min(auxAcZ),np.argmin(auxAcZ),np.max(auxAcZ),np.argmax(auxAcZ),robust.mad(auxAcZ),stattools.acf(auxAcZ).mean(),stattools.acf(auxAcX).std(),stattools.acovf(auxAcZ).mean(),stattools.acovf(auxAcZ).std(),skew(auxAcZ),kurtosis(auxAcZ),iqr(auxAcZ),
                                     mt.sqrt(np.mean(auxAcX)**2+np.mean(auxAcY)**2+np.mean(auxAcZ)**2),np.correlate(auxAcX,auxAcY),np.correlate(auxAcX,auxAcZ),np.correlate(auxAcZ,auxAcY),np.resize(np.fft.fftfreq(len(np.fft.fft(fich[:,1:]))),(100,))]];
                         aux = list(deepflatten(aux))
                         dados.append(aux)
                    except ValueError:  #raised if `y` is empty.
                        pass
                    y = y-1;
                    auxAcX = []
                    auxAcY = []
                    auxAcZ = []
                    valor = valor + 2;
                    
    with open("features.csv","w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(dados)
    return np.array(dados)

extract_features()