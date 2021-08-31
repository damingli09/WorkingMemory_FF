#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 15:18:10 2018
This simulation code tests how number of trials affects the FF estimation (variance of the estimator)
@author: Daming Li (daming.li@yale.edu)
"""


import numpy as np
import copy
import math
import matplotlib.pyplot as plt
import random as rd

rd.seed()

def spike_counter(mat,bins,spike_count):
    bin_width = bins[1]-bins[0]
    flag = 0
    for ind in range(len(mat)): # loop over trials
        if flag ==0:
            ind0 = ind
            flag = 1
        for spike_moment in mat[ind]: # loop over spike times in each trial
            i = int(math.floor(spike_moment/bin_width)) - int(bins[0]/bin_width)
            if (i>=0 and i<(len(bins)-1)):
                spike_count[ind-ind0,i] = spike_count[ind-ind0,i]+1
    for i in range(len(bins)-1):  # fill the columns full of zeros with nan
        if np.mean(spike_count[:,i]) == 0:
            spike_count[:,i] = np.full(spike_count.shape[0],np.nan)
            
def Fano_factor(spike_count): 
    num = np.var(spike_count, axis=0, ddof=1)
    denom = np.mean(spike_count, axis=0)
    num[np.where(denom==0)] = np.nan # no spike in that bin
    return np.divide(num,denom)

def FR(spike_count,bins): 
    bin_width = bins[1]-bins[0]
    return np.mean(spike_count,axis=0)*1000/bin_width           
    
def plot_telegraph(arr,end,H,L):
    temps = np.linspace(0,end,num=100000)
    tele = np.zeros(len(temps))
    for j in range(len(temps)):
        t = temps[j]
        for i in range(len(arr)):
            if arr[i] > t:
                flag = i
                break
        
        if flag%2 == 0:
            tele[j] = H
        else:
            tele[j] = L
    
    plt.figure(figsize=(8,1))        
    plt.plot(temps,tele,linewidth = 3) 
    ax = plt.gca()
    ax.axis('off')
    #plt.savefig('telegraph.pdf')
    plt.show()
        
def telegraph(muh,mul,H,L,end):
    #H = muh*(M*(1.0/muh+1.0/mul)-L/mul)
    n = []
    t = 0.0
    counter = 0
    #rd.seed()
    while t < end:
        if counter%2 == 0:
            #dt = -np.log(rd.uniform(0,1))/muh
            dt = 1000*rd.expovariate(muh)
            t = t + dt
            if t >= end:
                break
            n.append(t)
            counter = counter + 1
        else:
            #dt = -np.log(rd.uniform(0,1))/mul
            dt = 1000*rd.expovariate(mul)
            t = t + dt
            if t >= end:
                break
            n.append(t)
            counter = counter + 1
    
    n0 = copy.copy(n)    
    n0.insert(0,0)    
    n.append(end + 100) # this step is necesssary
    #plot_telegraph(n,end,H,L)
    
    t = 0.0
    train = []
    while t < end:
        #dt = -np.log(rd.uniform(0,1))/H
        dt = 1000*rd.expovariate(H)
        '''
        if dt < 2: # refractoriness
            #t = t + dt # should forward time step or not?
            continue
        '''
        if (t + dt) >= end:
            break
        u = rd.uniform(0,1)
        t = t + dt
        
        for i in range(len(n)): # is t in high state or low state?
            if n[i] > t:
                flag = i
                break
        
        if flag%2 == 0:
            rate = H
        else:
            rate = L
        
        if u > (rate/H):
            continue
        else:
            train.append(t)
    
    return train, n0

def spike_train_telegraph(Ntrials,muh,mul,H,L,end): # returns two arrays of arrays
    results = []
    n = []
    for i in range(Ntrials):
        '''
        plt.subplot(Ntrials, 1, Ntrials-i)
        if i==0 :
            plt.xlabel(r'Time (ms)',fontsize=18)
        else:
            plt.xticks([])
        '''
        train, n0 = telegraph(muh,mul,H,L,end)
        #plot_telegraph(n0,end,H,L)
        results.append(train)
        n.append(n0)
    '''
    plt.title(r'Firing rates (H=100Hz, L=10Hz)',fontsize=18)
    plt.savefig('firingrate.pdf')
    plt.show()
    '''
    return np.array(results),n # results are the spiking times, n are the telegraph level changing times
    
def FF(M,L,hlratio,taul,binwidth=0.25): # hlratio = tauh/taul, bin width = 0.25s
    H = M + (M-L)/hlratio  
    mul = 1.0/taul
    muh = 1.0/(hlratio*taul)
    tau = 1.0/(mul+muh)
    return 1.0 + 2.0*((H-L)**2)*mul*muh*(tau**2)*(np.exp(-binwidth/tau)-1+binwidth/tau)/(M*binwidth*((mul+muh)**2)) 

def FF_expressed_in_H(H,L,tauh,taul,bin_width=0.25): 
    hlratio = tauh/taul
    M = (H*tauh+L*taul)/(tauh+taul)
    return FF(M,L,hlratio,taul,bin_width)

N_array = np.arange(10,300,10)
Nrun = 10 # run Nrun times to see how centered and how spread the data is  
FF_array = np.zeros((Nrun,len(N_array)))
sample_mean_array = np.zeros(len(N_array)) # mean of the Nrun samples
sample_std_array = np.zeros(len(N_array)) # std of the Nrun samples

start = 0.0
end = 2000.0
bin_width = 250 # in ms, and should be much larger than the rate fluctuation scale
bins = np.linspace(start,end,num=(end-start)/bin_width+1)

L = 0.01 #0.01
H = 50.0 #150
tauh = 0.05
taul = 0.5
muh = 1.0/tauh
mul = 1.0/taul

FF_theory = FF_expressed_in_H(H,L,tauh,taul,0.001*bin_width)

for j in range(Nrun):  
    for i,Ntrials in enumerate(N_array):
        
        mat,n = spike_train_telegraph(Ntrials,muh,mul,H,L,2000)
        #rastor_plot(mat,n)
        spike_count = np.zeros((len(mat),len(bins)-1))
        spike_counter(mat,bins,spike_count)
        spike_count = spike_count[:,1:] # the beginning is always high, cut it off
        FR_evol = FR(spike_count,bins)
        FF_evol = Fano_factor(spike_count)
        #print np.mean(FR_evol[~np.isnan(FR_evol)])
        FF_evol = Fano_factor(spike_count)
        FF_array[j,i] = np.mean(FF_evol[~np.isnan(FF_evol)])
        #print np.mean(FF_evol[~np.isnan(FF_evol)])

for i in range(Nrun):      
    plt.plot(N_array,FF_array[i,:],'bo',alpha=0.25)  

sample_mean_array = np.mean(FF_array,axis=0)
sample_std_array = np.std(FF_array,axis=0)
plt.plot(N_array,sample_mean_array,'r-')
plt.plot(N_array,sample_mean_array-sample_std_array,'r-')
plt.plot(N_array,sample_mean_array+sample_std_array,'r-')
plt.xlabel("Number of trials",fontsize = 16)
plt.ylabel("Fano factor",fontsize = 16)

plt.plot(np.linspace(0,200,10000),FF_theory*np.ones(10000))
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig('effect_of_Ntrials.pdf')
plt.show()














