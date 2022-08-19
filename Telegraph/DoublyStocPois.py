"""
Created on Thu Feb 24 10:50:00 2022
This simulation code instantiates a doubly stochastic Poisson process with calculation of firing rate and Fano factor
@author: Daming Li (daming.li@yale.edu)
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

class DoublyStocPois(object):
    def __init__(self, rH, rL, tauH, tauL, duration=3):
        '''
        Instantiates a double stochastic Poisson process

        Parameters:
            rH: firing rate in high state, in sp/s
            rL: firing rate in low state, in sp/s
            tauH: mean duration in high state, in s
            tauL: mean duration in low state, in s
            duration: simulation length, in s
        '''

        assert rH >= rL
        self.rH = rH
        self.rL = rL
        self.tauH = tauH
        self.tauL = tauL
        self.duration = duration

    def run(self, refractoriness=False, ntrials=20):
        '''
        Simulates the model

        Parameters:
            refractoriness: if True, a 2 ms of refractoriness is added
            ntrials: number of trials

        Returns:
            telegraphs: trials of state switching timings, nested arrays
            timings: trials of spike trains, nested arrays
        '''

        telegraphs = []
        timings = []

        for i in range(ntrials):
            ### Simulate the telegraph process
            switchings = [0]  # start timing
            t = 0.0
            counter = 0
            while t < self.duration:
                if counter%2 == 0:  # high state, each process starts with high state
                    dt = np.random.exponential(self.tauH)
                else:  # low state
                    dt = np.random.exponential(self.tauL)
                t = t + dt
                if t >= self.duration:
                    break
                switchings.append(t)
                counter = counter + 1 
            switchings.append(self.duration)  # end timing
            telegraphs.append(switchings)  # append trial
            ### simulate the spike train
            '''
            t = 0.0
            train = []
            while t < self.duration:
                dt = np.random.exponential(1.0/self.rH)
                
                if refractoriness:
                    if dt < 2: # refractoriness
                        #t = t + dt 
                        continue
                
                if (t + dt) > self.duration:
                    break
                t = t + dt
                
                for i in range(len(switchings)): # determine if t is in high state or low state
                    if switchings[i] >= t:
                        flag = i
                        break
                
                if flag%2 == 0:
                    rate = self.rL
                else:
                    rate = self.rH
                
                u = np.random.uniform()
                if u > (rate/self.rH):  # Rejection and acceptance sampling
                    continue
                else:
                    train.append(t)
            timings.append(train)
            '''

            t = 0.0
            train = []
            for i in range(len(switchings)): # determine if t is in high state or low state
                if switchings[i] == 0: continue

                if i%2 == 0:
                    rate = self.rL
                else:
                    rate = self.rH

                while t <= switchings[i]:
                    dt = np.random.exponential(1.0/rate)
                    t += dt
                    if t <= switchings[i]:
                        train.append(t)

            timings.append(train)
                    
        self.telegraphs = telegraphs
        self.timings = timings
        
        return telegraphs, timings

    def spike_count(self, binsize=0.25):
        '''
        Calculate the spike counts of the simulation results

        Parameters:
            binsize: window size, in s
        '''
        
        nbins = int(self.duration/binsize)
        ntrials = len(self.timings)
        counts = np.zeros((ntrials, nbins))

        for i in range(ntrials):
            tmp = self.timings[i]
            for timing in tmp:
                if timing == 0: continue
                if timing == self.duration: break
                j = int(timing//binsize)
                counts[i,j] += 1
        
        self.counts = counts

        return counts

    def thStats(self, binsize=0.25):
        '''
        Theoretically computes mean firing rate and FF

        Parameters:
            binsize: bin size, in s

        Returns:
            FR: mean firing rate, in sp/s
            FF: Fano factor
        '''
        
        hlratio = self.tauH/self.tauL
        FR = (self.rH*self.tauH+self.rL*self.tauL)/(self.tauH+self.tauL)
        H = FR + (FR-self.rL)/hlratio  
        mul = 1.0/self.tauL
        muh = 1.0/(hlratio*self.tauL)
        tau = 1.0/(mul+muh)
        FF = 1.0 + 2.0*((H-self.rL)**2)*mul*muh*(tau**2)*(np.exp(-binsize/tau)-1+binsize/tau)/(FR*binsize*((mul+muh)**2))

        return FR, FF

    def empStats(self):
        '''
        Empirically computes mean firing rate and FF over time

        Returns:
            FR: mean firing rate, in sp/s
            FF: Fano factor
        '''
        
        FR = self.counts.mean(axis=0)  # mean spike count, not rate yet
        variance = np.var(self.counts, axis=0, ddof=1)
        FF = np.divide(variance,FR)
        binsize = self.duration/self.counts.shape[1]

        return FR/binsize, FF

    def raster_plot(self, trialNum=0):
        '''
        Visualize a telegraph and the corresponding spike train

        Parameters:
            trialNum: trial index
        '''

        temps = np.linspace(0, self.duration, num=100000)
        tele = np.zeros(len(temps))
        assert trialNum < len(self.telegraphs)
        switchtime = self.telegraphs[trialNum]  
        spikes = self.timings[trialNum] 

        for j in range(len(temps)):
            t = temps[j]
            for i in range(len(switchtime)):
                if switchtime[i] >= t:
                    flag = i
                    break
            
            if flag%2 == 0:
                tele[j] = self.rL
            else:
                tele[j] = self.rH
        
        plt.figure(figsize=(8,2))  
        plt.subplot(211)      
        plt.plot(temps,tele,linewidth = 3) 
        plt.xlim([0,self.duration])
        ax = plt.gca()
        ax.axis('off')

        plt.subplot(212)
        plt.plot(spikes,np.ones(len(spikes)),color="k",linestyle = 'None', marker="|",markersize = 45)
        plt.xlim([0,self.duration])
        '''
        markboxes = []
        for j in range(len(switchtime)//2):
            print(j)
            rect = patches.Rectangle((switchtime[2*j],-0.1),(switchtime[2*j+1]-switchtime[2*j]),0.2,linewidth=0.5,edgecolor='magenta',facecolor='magenta')
            markboxes.append(rect)
    
        pc = PatchCollection(markboxes,facecolor='magenta',alpha=0.5,edgecolor='None')
        ax.add_collection(pc)
        '''
        ax = plt.gca()
        ax.axis('off')

        plt.show()
