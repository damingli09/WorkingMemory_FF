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
    def __init__(self, df_spikes, df_stimulus, df_correct, stim_list=None, 
                start_fixation=0, end_fixation=1000, start_delay=1750, end_delay=4750, binsize=250):
        '''
        Instantiates a spike train analysis instance

        Parameters:
            df_spikes: a pandas dataframe with each column being a neuron, and entries being a list containing spike timings in ms
            df_stimulus: a pandas dataframe with each column being a neuron, and entries being the stimulus condition
            df_correct: a pandas dataframe with each column being a neuron, and binary entries indicating if that is a correct trial
            All these dataframes should match each other
            stim_list: list of all stimulus condition
            start_fixation: start time of the foreperiod, in ms
            end_fixation: end time of the foreperiod, in ms
            start_delay: start time of the delay, in ms
            end_delay: end time of the delay, in ms
            binsize: size of the analysis window, in ms
        '''

        self.allneurons = df_spikes.columns.tolist()  # list of all neurons' name

        if stim_list:
            self.stim_list = stim_list
        else:
            self.stim_list = df_stimulus.iloc[:,0].unique()[~np.isnan(df_stimulus.iloc[:,0].unique())]  # assume the first neuron contains all the stim conditions
        
        self.df_spikes = df_spikes
        self.df_stimulus = df_stimulus
        self.df_correct = df_correct
        self.start_fixation = start_fixation
        self.end_fixation = end_fixation
        self.start_delay = start_delay
        self.end_delay = end_delay
        self.binsize = binsize

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

        return 1000*(FR/self.binsize), FF

    def spike_count_under_a_stimulus(self, stimulus, neuron, start, end):
        '''
        Create the spike count matrix for a given neuron under a stimulus condition

        Parameters:
            stimulus: stimulus condition
            neuron: the neuron's name, a string
            start: start of the analysis window, in ms; for the delay period analysis this may be 250 ms after the cue offset
            end: end of the analysis window, in ms

        Returns:
            counts: spike count matrix (trials by time bin)
        '''

        series = self.df_spikes[neuron][(self.df_correct[neuron]==1.0)&(self.df_stimulus[neuron]==stimulus)]
        series.index = range(series.size) 
        counts = self.spike_count(series, start, end)  # trials by bins

        return counts

    def tuning(self, neuron, start, end, binsize=250, ANOVA=True, FR_Mat=None):
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

        self.bins = np.linspace(start,end,num=int((end-start)/binsize+1))
        pvalue_evolution = np.zeros(len(self.bins)-1) 

        if ANOVA:
            for i in range(len(self.bins)-1):   
                dic = {}
                for k in range(len(self.stim_list)):
                    stimulusk = self.spike_count_under_a_stimulus(self.stim_list[k], neuron, start, end)[:,i]
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

    def fit(self, neuron, start, end, binsize=250, ANOVA=True):
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

        self.bins = np.linspace(start,end,num=int((end-start)/binsize+1))

        for i in range(len(self.stim_list)): 
            stimulus = self.stim_list[i]  
            series = self.df_spikes[neuron][(self.df_correct[neuron]==1.0)&(self.df_stimulus[neuron]==stimulus)]
            series.index = range(series.size) 
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

    def single_neuron_stat(self, neuron):
        '''
        Summarize single neuron's behaviors in the task

        Parameters:
            neuron: name of the neuron

        Returns:
            FF_delay_preferred: mean FF during delay under the preferred stimulus
            FF_delay_least_preferred: mean FF during delay under the least preferred stimulus
            FF_foreperiod: mean FF during the foreperiod
            is_tuned: indicates if the neuron is well tuned during delay
            corFRFF, corFRFF_p: Pearson correlation (and p value) between mean firing rate and FF across stimulus conditions
            preferred_stim: the most preferred stimulus
            least_preferred_stim: the least preferred stimulus
            FF_prefered_bin: FF under the preferred stimulus at each tuned bin
            FF_least_prefered_bin: FF under the least preferred stimulus at each tuned bin
        '''

        ### Foreperiod
        _, FF_Mat, _, _ = self.fit(neuron, self.start_fixation, self.end_fixation, binsize=self.binsize, ANOVA=False)
        FF_foreperiod = np.nanmean(FF_Mat)

        ### Delay
        FR_Mat, FF_Mat, pvalues, is_tuned = self.fit(neuron, self.start_delay, self.end_delay, binsize=self.binsize, ANOVA=True)
        FF_temp = FF_Mat[:,pvalues < 0.05]
        FR_temp = FR_Mat[:,pvalues < 0.05]

        bin_FF_delay_preferred = []
        bin_FF_delay_least_preferred = []
        for i in range(FF_temp.shape[1]):
            stim_preferred = np.nanargmax(FR_temp[:,i]) # preferred stim for that bin
            stim_least_preferred = np.nanargmin(FR_temp[:,i])
            bin_FF_delay_preferred.append(FF_Mat[stim_preferred,i])
            bin_FF_delay_least_preferred.append(FF_Mat[stim_least_preferred,i])

        FF_Array = np.nanmean(FF_Mat[:,pvalues < 0.05],axis=1) # mean FF under each stimulus
        FR_Array = np.nanmean(FR_Mat[:,pvalues < 0.05],axis=1) # mean FR under each stimulus
        stim_preferred = np.nanargmax(FR_Array)
        stim_least_preferred = np.nanargmin(FR_Array)
        FF_delay_preferred = FF_Array[stim_preferred]
        FF_delay_least_preferred = FF_Array[stim_least_preferred]
        bin_FF_delay_preferred = np.array(bin_FF_delay_preferred)
        bin_FF_delay_least_preferred = np.array(bin_FF_delay_least_preferred)

        ### cor(FR, FF)
        if np.shape(FR_temp)[1] == 0:
            corFRFF, corFRFF_p = np.nan, np.nan
        else:
            FF_Array = FF_temp[:,0]
            FR_Array = FR_temp[:,0]
            
            for i in range(np.shape(FR_temp)[1]):
                if i == 0:
                    continue
                FF_Array = np.append(FF_Array,FF_temp[:,i])
                FR_Array = np.append(FR_Array,FR_temp[:,i])

            corFRFF, corFRFF_p = stats.pearsonr(FF_Array[~np.isnan(FF_Array)], FR_Array[~np.isnan(FF_Array)])

        return FF_delay_preferred, FF_delay_least_preferred, FF_foreperiod, is_tuned, corFRFF, corFRFF_p, stim_preferred, stim_least_preferred, bin_FF_delay_preferred, bin_FF_delay_least_preferred
