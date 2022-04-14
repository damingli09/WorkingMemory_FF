"""
Created on Thu Feb 24 10:50:00 2022
This simulation code instantiates a doubly stochastic Poisson process with calculation of firing rate and Fano factor
@author: Daming Li (daming.li@yale.edu)
"""

import numpy as np
import pandas as pd
import math
from scipy import stats
import matplotlib.pyplot as plt

class SpikeTrainAnalysis(object):
    def __init__(self, df_spikes, df_stimulus, df_correct, stim_list=None):
        '''
        Instantiates a spike train analysis instance

        Parameters:
            df_spikes: a pandas dataframe with each column being a neuron, and entries being a list containing spike timings in ms
            df_stimulus: a pandas dataframe with each column being a neuron, and entries being the stimulus condition
            df_correct: a pandas dataframe with each column being a neuron, and binary entries indicating if that is a correct trial
            All these dataframes should match each other
            stim_list: list of all stimulus condition
        '''

        self.allneurons = df_spikes.columns.tolist()  # list of all neurons' name

        if stim_list:
            self.stim_list = stim_list
        else:
            self.stim_list = df_stimulus.iloc[:,0].unique()[~np.isnan(df_stimulus.iloc[:,0].unique())]  # assume the first neuron contains all the stim conditions
        
        self.df_spikes = df_spikes
        self.df_stimulus = df_stimulus
        self.df_correct = df_correct

    def spike_count(self, series, start, end):
        '''
        Calculate the spike counts of the simulation results

        Parameters:
            series: pandas series for a given neuron
            start: start of the analysis window, in ms
            end: end of the analysis window, in ms
        '''
        
        counts = np.zeros((series.size,len(self.bins)-1))
        self.binsize = self.bins[1] - self.bins[0]

        for i in range(series.size):
            tmp = series[i]
            for timing in tmp:
                if timing <= start: continue
                if timing >= end: break
                j = int((timing-start)//self.binsize)
                counts[i,j] += 1

        return counts

    def empStats(self, counts):
        '''
        Empirically computes mean firing rate and FF over time

        Parameters:
            counts: spike counts matrix, trial by bins

        Returns:
            FR: mean firing rate in each bin, in sp/s
            FF: Fano factor in each bin
        '''
        
        FR = counts.mean(axis=0)  # mean spike count, not rate yet
        variance = np.var(counts, axis=0, ddof=1)
        FF = np.divide(variance,FR)

        return FR/self.binsize, FF

    def spike_count_uner_a_stimulus(self, stimulus, neuron, start, end, binsize):
        '''
        Create the spike count matrix for a given neuron under a stimulus condition

        Parameters:
            stimulus: stimulus condition
            neuron: the neuron's name, a string
            start: start of the analysis window, in ms; for the delay period analysis this may be 250 ms after the cue offset
            end: end of the analysis window, in ms
            binsize: window size, 250 ms

        Returns:
            counts: spike count matrix (trials by time bin)
        '''

        series = self.df_spikes[neuron][(self.df_correct[neuron]==1.0)&(self.df_stimulus[neuron]==stimulus)]
        series.index = range(series.size) 
        counts = self.spike_count(series, start, end)  # trials by bins

        return counts

    def tuning(self, neuron, start, end, binsize=0.25, ANOVA=True, FR_Mat=None):
        '''
        Test if a neuron at a time bin is well tuned

        Parameters:
            neuron: the neuron's name, a string
            start: start of the analysis window, in ms; for the delay period analysis this may be 250 ms after the cue offset
            end: end of the analysis window, in ms
            binsize: window size, 250 ms
            ANOVA: if True, F-test is used (such as ODR and MNM), otherwise linear regression is applied (such as VDD)
            FR_Mat: mean firing rate under each stimulus condition at each time bin

        Returns:
            pvalues: an array of p values at each time bin
        '''

        self.bins = np.linspace(start,end,num=(end-start)/binsize+1)
        pvalue_evolution = np.zeros(len(self.bins)-1) 

        if ANOVA:
            for i in range(len(self.bins)-1):   
                dic = {}
                for k in range(len(self.stim_list)):
                    stimulusk = self.spike_count_under_a_stimulus(self.stim_list[k], neuron, start, end, binsize)[:,i]
                    if np.mean(stimulusk) > 0:
                        dic[str(k)] = stimulusk
                if len(dic.keys()) > 1:
                    _,p = stats.f_oneway(*dic.values())
                    pvalue_evolution[i] = p    
                else:
                    pvalue_evolution[i] = np.nan
        else:
            for i in range(len(pvalue_evolution)):
                x = self.stim_list
                y = FR_Mat[:,i]
                _, _, _, p_value, _ = stats.linregress(x,y)
                pvalue_evolution[i] = p_value

        return pvalue_evolution

    def consecutive_indices(self, array):
        '''
        Test if there are at least two consecutive bins with well-tuned activity
        '''

        consecutive = False
        for element in array:
            if ((element+1) in array):
                consecutive = True
                break

        return consecutive

    def fit(self, neuron, start, end, binsize=0.25, ANOVA=True):
        '''
        Compute mean firing rate, Fano factor, and run F test for a given neuron in a task epoch.

        Parameters:
            neuron: the neuron's name, a string
            start: start of the analysis window, in ms; for the delay period analysis this may be 250 ms after the cue offset
            end: end of the analysis window, in ms
            binsize: window size, 250 ms
            ANOVA: if True, F-test is used (such as ODR and MNM), otherwise linear regression is applied (such as VDD)
        
        Returns:
            FR_Mat: mean firing rate matrix (stimulus by time bins)
            FF_Mat: FF matrix (stimulus by time bins)
            pvalues: p value of the tuning test at each time bin
            is_tuned: indicates if this is a well-tuned neuron
        '''

        self.bins = np.linspace(start,end,num=(end-start)/binsize+1)

        for i in range(len(self.stim_list)): 
            stimulus = self.stim_list[i]  
            series = self.df_spikes[neuron][(self.df_correct[neuron]==1.0)&(self.df_stimulus[neuron]==stimulus)]
            counts = self.spike_count(series, start, end)  # trials by bins
            FR_evolution, FF_evolution = self.empStats(counts)

            if i==0:
                FR_Mat = FR_evolution
                FF_Mat = FF_evolution
            else:
                FR_Mat = np.vstack([FR_Mat, FR_evolution])
                FF_Mat = np.vstack([FF_Mat, FF_evolution])

        pvalues = self.tuning(neuron, start, end, binsize, ANOVA, FR_Mat)
        well_tuned_indices = np.where(pvalues < 0.05)[0]
        is_tuned = self.consecutive_indices(well_tuned_indices)

        return FR_Mat, FF_Mat, pvalues, is_tuned
