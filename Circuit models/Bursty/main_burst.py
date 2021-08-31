"""
A model of decision making and working memory from X-J Wang (Neuron 2002) implemented in the Brian simulator,
with adaptation and facilitation added.
Contact: Daming Li (daming.li@yale.edu)
"""

import matplotlib
matplotlib.use('Agg') # to generate figures using the cluster
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("/net/murray/nhl8/Modules/brian-1.4.1/")
import numpy as np
from numpy.random import seed
from pylab import *
import argparse
#import brian_no_units           #Note: speeds up the code
from brian2 import *
import analysis

## For optimization, need to update for B2.
# set_global_preferences(useweave=True,usecodegen=True,usecodegenweave=True,usecodegenstateupdate=True,usenewpropagate=True,usecodegenthreshold=True,gcc_options=['-ffast-math','-march=native'])

#-----------------------------------------------------------------------------------------
### Load default parameters ###
execfile('parameters_burst.py')
# #Pass arguments from autojob_mercer.py. Note that all but strings need to be eval() first.
# savepath    = sys.argv[1]	# Start from 1, 0 is about the structure of the passes arguments or sth
# parameter_1_name = sys.argv[2]
# parameter_1_unit = eval(sys.argv[3])
# parameter_1 = eval(sys.argv[4])
# #Repeat sys.argv[2-4] for multiple parameters
suffix = 'Burst'

# Dynamical equations
eqs_e = '''
dv/dt = (-g_KCa_e*(v-E_K_e)*Ca_e-gl_e*(v-El_e)-g_gaba_e*s_gaba*(v-E_gaba)-g_ampa_e*s_ampa*(v-E_ampa)-g_nmda_e*s_tot*(v-E_nmda)/(1+b*exp(-a*v)))/Cm_e: volt (unless refractory)
ds_gaba/dt = -s_gaba/t_gaba : 1
ds_ampa/dt = -s_ampa/t_ampa : 1
ds_nmda/dt = -s_nmda/t_nmda+alpha*x*(1-s_nmda) : 1
dx/dt = -x/t_x : 1
dF_nmda/dt = -F_nmda / t_F_nmda : 1
dCa_e/dt = -Ca_e/t_Ca :1
s_tot : 1
'''
eqs_i = '''
dv/dt = (-gl_i*(v-El_i)-g_gaba_i*s_gaba*(v-E_gaba)-g_ampa_i*s_ampa*(v-E_ampa)-g_nmda_i*s_tot*(v-E_nmda)/(1+b*exp(-a*v)))/Cm_i: volt (unless refractory)
ds_gaba/dt = -s_gaba/t_gaba : 1
ds_ampa/dt = -s_ampa/t_ampa : 1
s_tot : 1
'''

Ntrials = 1
for ii in range(Ntrials):
    print 'run '+str(ii)

    Pe = NeuronGroup(NE, eqs_e, threshold='v>Vt_e', reset='v=Vr_e', refractory=tr_e, order=2, clock=simulation_clock)
    Pe.v = El_e
    Pe.s_ampa = 0
    Pe.s_gaba = 0
    Pe.s_nmda = 0
    Pe.x = 0
    Pe.F_nmda = 0
    Pe.Ca_e = 0
    
    Pi = NeuronGroup(NI, eqs_i, threshold='v>Vt_i', reset='v=Vr_i', refractory=tr_i, order=2, clock=simulation_clock)
    Pi.v = El_i
    Pi.s_ampa = 0
    Pi.s_gaba = 0
    
    # create 3 excitatory subgroups: 1 & 2 are selective to motion direction, 0 is not
    Pe0 = Pe[:N0]
    Pe1 = Pe[N0:(N0+N1)]
    Pe2 = Pe[(N0+N1):]

    # external Poisson input
    PGi = PoissonGroup(NI, fext, clock=simulation_clock)
    PG0 = PoissonGroup(N0, fext, clock=simulation_clock)
    PG1 = PoissonGroup(N1, stimulus1, clock=simulation_clock)
    PG2 = PoissonGroup(N2, stimulus2, clock=simulation_clock)
    # PG1 = NeuronGroup(N1, '', threshold='rand()<stimulus1(t)*dt_default', clock=simulation_clock)
    # PG2 = NeuronGroup(N2, '', threshold='rand()<stimulus2(t)*dt_default', clock=simulation_clock)
    
    
    selfnmda = Synapses(Pe, Pe, 'w:1', on_pre='''x+=300.0*F_nmda 
                                                F_nmda += alpha_F_nmda*(1.-F_nmda)''', clock=simulation_clock) # Update NMDA gating variables of E-cells (pre-synaptic)
    selfnmda.connect(j='i')
    selfnmda.w = 1.
    selfnmda.delay = '0.5*ms'

    # Poisson Synapses
    Cp1 = Synapses(PG1, Pe1, 'w:1', on_pre='s_ampa_post+=w', clock=simulation_clock) # Update external AMPA gating variables of Group 1 E-cells (pre-synaptic)
    Cp1.connect(j='i')
    Cp1.w = wext_e
    Cp2 = Synapses(PG2, Pe2, 'w:1', on_pre='s_ampa_post+=w', clock=simulation_clock) # Update external AMPA gating variables of Group 2 E-cells (pre-synaptic)
    Cp2.connect(j='i')
    Cp2.w = wext_e
    Cp0 = Synapses(PG0, Pe0, 'w:1', on_pre='s_ampa_post+=w', clock=simulation_clock) # Update external AMPA gating variables of non-selective E-cells (pre-synaptic)
    Cp0.connect(j='i')
    Cp0.w = wext_e
    Cpi = Synapses(PGi, Pi, 'w:1', on_pre='s_ampa_post+=w', clock=simulation_clock) # Update external AMPA gating variables of I-cells (pre-synaptic)
    Cpi.connect(j='i')
    Cpi.w = wext_i

    # Recurrent Synapses (non-NMDA)
    Cie = Synapses(Pi, Pe, 'w:1', on_pre='s_gaba_post+=w', clock=simulation_clock) # Update GABA gating variables of all E-cells (pre-synaptic)
    Cie.connect()
    Cie.w = 1.
    Cie.delay = '0.5*ms'
    Cii = Synapses(Pi, Pi, 'w:1', on_pre='s_gaba_post+=w', clock=simulation_clock) # Update GABA gating variables of I-cells (pre-synaptic)
    Cii.connect()
    Cii.w = 1.
    Cii.delay = '0.5*ms'
    Cei = Synapses(Pe, Pi, 'w:1', on_pre='s_ampa_post+=w', clock=simulation_clock) # Update AMPA gating variables of I-cells (pre-synaptic)
    Cei.connect()
    Cei.w = 1.
    Cei.delay = '0.5*ms'
    C00 = Synapses(Pe0, Pe0, 'w:1', on_pre='s_ampa_post+=w', clock=simulation_clock) # Update AMPA gating variables within non-selective E-cells (pre-synaptic)
    C00.connect()
    C00.w = 1.
    C00.delay = '0.5*ms'
    C10 = Synapses(Pe1, Pe0, 'w:1', on_pre='s_ampa_post+=w', clock=simulation_clock) # Update AMPA gating variables from group 1 to non-selective E-cells (pre-synaptic)
    C10.connect()
    C10.w = 1.
    C10.delay = '0.5*ms'
    C20 = Synapses(Pe2, Pe0, 'w:1', on_pre='s_ampa_post+=w', clock=simulation_clock) # Update AMPA gating variables from group 2 to non-selective E-cells (pre-synaptic)
    C20.connect()
    C20.w = 1.
    C20.delay = '0.5*ms'
    C01 = Synapses(Pe0, Pe1, 'w:1', on_pre='s_ampa_post+=w', clock=simulation_clock) # Update AMPA gating variables from non-selective to group 1 E-cells (pre-synaptic)
    C01.connect()
    C01.w = wm
    C01.delay = '0.5*ms'
    C11 = Synapses(Pe1, Pe1, 'w:1', on_pre='s_ampa_post+=w', clock=simulation_clock) # Update AMPA gating variables within group 1 E-cells (pre-synaptic)
    C11.connect()
    C11.w = wp
    C11.delay = '0.5*ms'
    C21 = Synapses(Pe2, Pe1, 'w:1', on_pre='s_ampa_post+=w', clock=simulation_clock) # Update AMPA gating variables from group 2 to non-selective E-cells (pre-synaptic)
    C21.connect()
    C21.w = wm
    C21.delay = '0.5*ms'
    C02 = Synapses(Pe0, Pe2, 'w:1', on_pre='s_ampa_post+=w', clock=simulation_clock) # Update AMPA gating variables from non-selective to group 2 E-cells (pre-synaptic)
    C02.connect()
    C02.w = wm
    C02.delay = '0.5*ms'
    C12 = Synapses(Pe1, Pe2, 'w:1', on_pre='s_ampa_post+=w', clock=simulation_clock) # Update AMPA gating variables from group 1 to group 2 E-cells (pre-synaptic)
    C12.connect()
    C12.w = wm
    C12.delay = '0.5*ms'
    C22 = Synapses(Pe2, Pe2, 'w:1', on_pre='s_ampa_post+=w', clock=simulation_clock) # Update AMPA gating variables within group 2 E-cells (pre-synaptic)
    C22.connect()
    C22.w = wp
    C22.delay = '0.5*ms'

    Cee_Ca = Synapses(Pe, Pe, 'w:1', on_pre='Ca_e+=w', method='Euler', clock=simulation_clock) # Update external AMPA gating variables of E-cells (pre-synaptic)
    Cee_Ca.connect(j='i')
    Cee_Ca.w=0.324 # 0.1 the lower the more persistent, the higher the more silent


    # Calculate NMDA contributions (Post-synaptic)
    @network_operation(clock=simulation_clock, when='start')
    def update_nmda():
        s_NMDA1 = np.sum(Pe1.s_nmda)
        s_NMDA2 = np.sum(Pe2.s_nmda)
        s_NMDA0 = np.sum(Pe0.s_nmda)
        Pe1.s_tot = (wp*s_NMDA1+wm*s_NMDA2+wm*s_NMDA0)
        Pe2.s_tot = (wm*s_NMDA1+wp*s_NMDA2+wm*s_NMDA0)
        Pe0.s_tot = (s_NMDA1+s_NMDA2+s_NMDA0)
        Pi.s_tot  = (s_NMDA1+s_NMDA2+s_NMDA0)
    
    # initiating monitors
    monit_P1 = SpikeMonitor(Pe, record=True)
    #monit_P2 = SpikeMonitor(Pe2, record=True)
    
    # plot and save data 
    dt_state_mon = 0.5 *ms
    # state_mon_clock = Clock(dt=dt_state_mon)
    # mon_states_e = StateMonitor(Pe, ['v', 'x_nmda', 's_tot', 's_nmda', 's_ext'], record=True, clock=state_mon_clock)
    mon_states_e = StateMonitor(Pe, ['v', 'x', 'F_nmda','s_nmda','Ca_e'], record=[1200], dt=dt_state_mon)
    # mon_states_e_nmda = StateMonitor(Pe, ['s_tot', 's_nmda', 's_ext'], record=True, clock=state_mon_clock)
    # print mon_states_e_nmda.s_ext

    # Run Simulations
    net = Network(Pe,Pe0,Pe1,Pe2,Pi,PG0,PG1,PG2,PGi,selfnmda,Cp0,Cp1,Cp2,Cpi,Cie,Cii,Cei,C00,C10,C20,C01,C11,C21,C02,C12,C22,Cee_Ca,update_nmda,monit_P1,mon_states_e)
    net.run(simtime, report='text')
    
    # Save data
    np.save('spikes_e_'+suffix+'_cue_'+str(coherence)+'_run_'+str(ii)+'.npy',np.column_stack((monit_P1.i, monit_P1.t)))
    #np.save('spikes_e_'+suffix+'_cue_'+str(coherence)+'_run_'+str(ii)+'.npy',np.column_stack((monit_P2.i, monit_P2.t)))

    #Saving states data
    # mon_states_e_x_mean = np.asarray(mon_states_e.x_nmda).mean(axis=0)
    mon_states_e_v_mean = (mon_states_e.v).mean(axis=0)
    mon_states_e_x_mean = (mon_states_e.x).mean(axis=0)
    mon_states_e_F_mean = (mon_states_e.F_nmda).mean(axis=0)
    mon_states_e_Ca_mean = (mon_states_e.Ca_e).mean(axis=0)
    np.save('state_e_x_'+suffix+'_cue_'+str(coherence)+'_run_'+str(ii)+'.npy', np.column_stack((mon_states_e.t, mon_states_e_v_mean, mon_states_e_x_mean,mon_states_e_F_mean,mon_states_e_Ca_mean)))
    
    
    #Testing for Wang2004
    #Saving State Variable terms, assorted by compartment.
    mon_states_e_nmda_s_nmda_mean = (mon_states_e.s_nmda).mean(axis=0)
    np.save('state_e_NMDA_'+suffix+'_cue_'+str(coherence)+'_run_'+str(ii)+'.npy', np.column_stack((mon_states_e.t, mon_states_e_nmda_s_nmda_mean)))


########################################################################################################################
### Make figures
dat = np.load('spikes_e_Burst_cue_90.0_run_0.npy')

plt.figure(figsize=(16,10))
for i in range(1600):
    spikes = dat[:,1][dat[:,0]==i]
    #print spikes
    plt.plot(spikes,i*np.ones(len(spikes)),'r.')
plt.savefig('figures/tCa'+str(t_Ca)+'t_F'+str(t_F_nmda)+'Cee_Ca'+str(Cee_Ca.w)+'.pdf')

### Run Analysis here. Copied from cluster.
if False:
    # Define needed params instead of loading parameter.py, to save time.
    N1 = 240
    N2 = 240
    simtime = 5.0*second     # total simulation time [ms]
    dt_psth = 0.001*second

    #Pre define t_vec_list using the same expression in psth function by John.
    # tpts_0 = int(np.round((ts_stop-ts_start)/dt_psth))+1
    # t_vec_list = np.linspace(ts_start, ts_stop, tpts_0)
    tpts_0 = int(np.round((simtime-0.*second)/dt_psth))+1
    t_vec_list = np.linspace(0., simtime/second, tpts_0)
    # print(tpts_0)
    # print(len(t_vec_list))






    ### Summarize just simulated data
    filename_spikes_Pe1  = ("spikes_s1.dat")   # Set file name for soma spikes.
    filename_spikes_Pe2  = ("spikes_s2.dat")   # Set file name for soma spikes.
    #Allow empty spike-sets to be evaluataed as 0 spike rates...
    # if os.stat(savepath+filename_spikes_Pe1).st_size == 0:
    #     r_Pe1_list[:, i_parameter_1_list, i_variable_manual, i_set] = np.zeros(tpts_0)
    # elif os.stat(savepath+filename_spikes_Pe1).st_size > 0:
    if True:
        spikes_Pe1_temp  = analysis.load_raster_spikes(filename_spikes_Pe1)
        r_Pe1_temp       = analysis.psth(spikes_Pe1_temp[:,1], t_vec=t_vec_list, t_start=0., t_end=simtime/second, filter_fn=analysis.filter_exp)/N1
        # r_Pe1_temp       = analysis.psth(spikes_Pe1_temp[:,1], t_vec=t_vec_list, t_start=0., t_end=simtime/second, filter_fn=analysis.filter_gauss)/N1
    #    r_Pe1_temp       = analysis.psth(spikes_Pe1_temp[:,1], t_vec=t_vec_list, t_start=0., t_end=simtime, filter_fn=analysis.filter_gauss)/N1
        # r_Pe1_list[:, i_parameter_1_list, i_variable_manual, i_set] = r_Pe1_temp                               # Note that these rates durectly uses Psth and is smooth...
    # if os.stat(savepath+filename_spikes_Pe2).st_size == 0:
    #     r_Pe2_list[:, i_parameter_1_list, i_variable_manual, i_set] = np.zeros(tpts_0)
    # elif os.stat(savepath+filename_spikes_Pe2).st_size > 0:

    # ### Thiago's way
    # num_bins = 5000 # bin = 1*ms
    # name = savepath+filename_spikes_Pe1
    # with open('%s' % (name), 'r') as f:
    #     spike_times = []
    #     for line in f:
    #         current_line = line.split(",")
    #         spike_times.append(float(current_line[1]))
    #         freq, dummy = np.histogram(spike_times, bins=num_bins, range=None, normed=False, weights=None)
    #         freq = freq*1000/240.0
    #     r_Pe1_temp = analysis.smoothListGaussian(freq,150)


    if True:
        spikes_Pe2_temp  = analysis.load_raster_spikes(filename_spikes_Pe2)
        r_Pe2_temp       = analysis.psth(spikes_Pe2_temp[:,1], t_vec=t_vec_list, t_start=0., t_end=simtime/second, filter_fn=analysis.filter_exp)/N2
        # r_Pe2_temp       = analysis.psth(spikes_Pe2_temp[:,1], t_vec=t_vec_list, t_start=0., t_end=simtime/second, filter_fn=analysis.filter_gauss)/N2
    #    r_Pe2_temp       = analysis.psth(spikes_Pe2_temp[:,1], t_vec=t_vec_list, t_start=0., t_end=simtime, filter_fn=analysis.filter_gauss)/N2
        # r_Pe2_list[:, i_parameter_1_list, i_variable_manual, i_set] = r_Pe2_temp                               # Note that these rates durectly uses Psth and is smooth...

    # ### Thiago's way
    # name = savepath+filename_spikes_Pe2
    # with open('%s' % (name), 'r') as f:
    #     spike_times = []
    #     for line in f:
    #         current_line = line.split(",")
    #         spike_times.append(float(current_line[1]))
    #         freq, dummy = np.histogram(spike_times, bins=num_bins, range=None, normed=False, weights=None)
    #         freq = freq*1000/240.0
    #     r_Pe2_temp = analysis.smoothListGaussian(freq,150)





    win_12=0
    RT_win=0.
    # Define winner as the first of the 2 group who reaches 15Hz first. Supposedly psth should smooth the distributions well enough to prevent noise from making a difference...
    for ii in xrange(len(r_Pe1_temp)):
        if r_Pe1_temp[ii] >= 15.0 :
           win_12 = 1
           RT_win = t_vec_list[ii]
           break
        elif r_Pe2_temp[ii] >= 15.0 :
           win_12 = 2
           RT_win = t_vec_list[ii]
           break
        else :
            win_12 = 0      #Should also change RT? fix later...




    ## Save various plots and data.
    #Save Scatter Plots for P cells.
    # if not os.path.exists(savepath+'spikes_scatter_E12.pdf'):      						  #Create path if it does not exist
    if True:
        fig1 = plt.figure(figsize=(8,10.5))
        n_skip=8 # sparsen scatter plots
        ### Spike rastergram for E1 ###
        ax11 = fig1.add_subplot(311)
        ax11.scatter(spikes_Pe1_temp[:,1][::n_skip],spikes_Pe1_temp[:,0][::n_skip],\
                    marker='.',c='red',edgecolor='none',alpha=0.5)
        # Note: [::n] means jump to the nth element, skipping the n-1 ones...
        # ax11.set_ylim(0,360)
        ax11.set_xlim(0,simtime/second)
        ax11.set_ylabel('Neuron label, E1 (deg)')
        ax11.set_xlabel('Time (s)')

        ### Spike rastergram for E2 ###
        ax12 = fig1.add_subplot(312)
        ax12.scatter(spikes_Pe2_temp[:,1][::n_skip],spikes_Pe2_temp[:,0][::n_skip],\
                    marker='.',c='blue',edgecolor='none',alpha=0.5)
        # Note: [::n] means jump to the nth element, skipping the n-1 ones...
        # ax12.set_ylim(0,360)
        ax12.set_xlim(0,simtime/second)
        ax12.set_ylabel('Neuron label, E2 (deg)')
        ax12.set_xlabel('Time (s)')

        ### Firing rate profiles for P cells###
        ax13 = fig1.add_subplot(313)
        ax13.plot(t_vec_list,r_Pe1_temp,c='red' ,lw=2,label='E1')
        ax13.plot(t_vec_list,r_Pe2_temp,c='blue',lw=2,label='E2')
        ax13.set_xlabel('Time (s)')
        ax13.set_ylabel('Population Firing rate (Hz)')
        ax13.set_xlim(0,simtime/second)
        # ax13.set_ylim(0,)
        ax13.legend(frameon=False    )
        fig1.savefig('spikes_scatter_E12.pdf')


    np.savetxt("r_smooth_E1.txt", r_Pe1_temp)
    np.savetxt("r_smooth_E2.txt", r_Pe2_temp)
    np.savetxt("win12_RTwin.txt", (win_12, RT_win))
