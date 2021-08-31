% SpikingNetworkModel.m: Spiking network for parametric working memory 
% based on negative derivative feedback.
% Created by Sukbin Lim
%
% This program reproduces the experiment in Fig. 5 in Lim and Goldman
% (2013), Nature Neuroscience
%
% Warning: Simulation in matlab program is slow for a large size of
% network which is required for strong corrective feedback

clear all;clc;close all
%% Parameters
% integration duration and time step 
dt = .1;                        % Time step (ms)

Tinit = round(1000/dt);          % # of steps to initialize the state
Tstim = round(500/dt);          % # of steps during stimulus presentation
Tmemory = round(4000/dt);       % # of steps during delay period

T = Tinit + Tstim + Tmemory;    % # of total integration steps

% size of population and connectivity probability
scale = 20;
Nex = 16000/scale;        Nin = 4000/scale;         NAll = Nex + Nin;      
Next = NAll;        

scale_p = 5;
ConnProb = 0.1*scale_p;  

% firing rate of external population
InitRate = 0.4;      % firing rate of ext. pop. during initialization 0.4, 200 ms
InputRate = 0.309*0.2;    % firing rate of ext. pop. during stimulus presentation 0.309, 200ms

% strength of recurrent connectivity
JEE = 0.3750*sqrt(scale/scale_p);   % E to E
JEI = -1*sqrt(scale/scale_p);       % I to E
JIE = 0.3750*sqrt(scale/scale_p);   % E to I
JII = -1*sqrt(scale/scale_p);       % I to I

% strength of feedforward connectivity
JEO = 0.0224*sqrt(scale/scale_p);    % input J to exc
JIO = 0*sqrt(scale/scale_p);         % input J to inh

% recurrent connection matrix 
ConnMatrix = spalloc(NAll, NAll, round(ConnProb*NAll^2*1.1)); 
for j = 1:NAll
    Columnj = ceil(rand(NAll, 1) - (1-ConnProb));
    Columnj(j) = 0; % Remove self-connection
    ConnMatrix(:, j) = Columnj;
end

% feedforward connection matrix
clear Columnj
InputConnMatrix = spalloc(NAll, Next, round(ConnProb*NAll*Next*1.1));
for j = 1:Next
    Columnj = ceil(rand(NAll, 1) - (1-ConnProb));    
    InputConnMatrix(:, j) = Columnj;
end

% parameters for individual neurons
AbsRefractory = round(2/dt);    % absolute refractory period [time steps]

% Thresholds
ThreshEx = 1;   ThreshIn = 1;       

% Reset value
ResetEx = 0.4;  ResetIn = 0.4;                 

% Membrane time constants
TauEx = 20;     TauIn = 10;      

% parameters for synaptic currents
TauSynO   = 100;    % Time constant of filtering of external input
TauSynEE1 = 150;    % Slow decay time constant of E-E synaptic connection 
TauSynEE2 = 50;     % Fast decay time constant of E-E synaptic connection
TauSynIE1 = 45;     % Slow decay time constant of E-I synaptic connection
TauSynIE2 = 20;     % Fast decay time constant of E-I synaptic connection
TauSynEI  = 10;      % Decay time constant of I-E synaptic connection, 10 in paper, 5 in code
TauSynII  = 10;      % Decay time constant of I-I synaptic connection, 10 in paper, 5 in code

pEE = 0.5;          % Fraction of slow currents in E-E synaptic connection      
pIE = 0.2;          % Fraction of slow currents in E-I synaptic connection

SyndecayO  = exp(-dt / TauSynO);
SyndecayEE1 = exp(-dt / TauSynEE1); 
SyndecayEE2 = exp(-dt / TauSynEE2); 
SyndecayIE1 = exp(-dt / TauSynIE1);  
SyndecayIE2 = exp(-dt / TauSynIE2);  
SyndecayEI = exp(-dt / TauSynEI);  
SyndecayII = exp(-dt / TauSynII);  

Ntrials = 1;

for ii = 1:Ntrials
    %% Initializations:
    New.MemPotEx = zeros(Nex, 1);  
    New.MemPotIn = zeros(Nin, 1);  
    New.SynO  = zeros(Next,1);
    New.SynEE1 = zeros(Nex, 1);
    New.SynEE2 = zeros(Nex, 1);
    New.SynIE1 = zeros(Nex, 1);     
    New.SynIE2 = zeros(Nex, 1);     
    New.SynEI = zeros(Nin, 1);     
    New.SynII = zeros(Nin, 1);    

    Old.MemPotEx = zeros(Nex, 1);  
    Old.MemPotIn = zeros(Nin, 1);  
    Old.SynO  = zeros(Next, 1);
    Old.SynEE1 = zeros(Nex, 1);    
    Old.SynEE2 = zeros(Nex, 1);    
    Old.SynIE1 = zeros(Nex, 1); 
    Old.SynIE2 = zeros(Nex, 1); 
    Old.SynEI = zeros(Nin, 1);    
    Old.SynII = zeros(Nin, 1);   

    Temp.Ex = zeros(Nex, 1);
    Temp.In = zeros(Nin, 1);

    Spikes.Ex = sparse(Nex, round(T)+1); 
    Spikes.In = sparse(Nin, round(T)+1); 

    Index.Ex = sparse(Nex, 1); 
    Index.In = sparse(Nin, 1); 

    Tspikes.Ex = sparse(Nex, 1);
    Tspikes.In = sparse(Nin, 1);

    Activity.Ex = zeros(1,round(T)+1);    % Network activity for exc population
    Activity.In = zeros(1,round(T)+1);    % Network activity for inh population

    % variables related to refractoriness
    Refr.Ex = zeros(Nex, 1);    % Refractory states
    Refr.In = zeros(Nin, 1);    % Refractory states
    Temp.MemPotEx = zeros(Nex, 1);
    Temp.MemPotIn = zeros(Nin, 1);
    HistMemPot.Ex = zeros(Nex,AbsRefractory);
    HistMemPot.In = zeros(Nex,AbsRefractory);
    HistSpikes.Ex = zeros(Nex,AbsRefractory);
    HistSpikes.In = zeros(Nex,AbsRefractory);

    Stor.MemPotEx = zeros(Nex,500); 
    Stor.MemPotIn = zeros(Nin,500); 
    Stor.SynEE1 = zeros(Nex,500); 
    Stor.SynEE2 = zeros(Nex,500); 
    Stor.SynEI = zeros(Nin,500);  
    Stor.SynIE1 = zeros(Nex,500); 
    Stor.SynIE2 = zeros(Nex,500); 
    Stor.SynII = zeros(Nin,500); 
    %%
    Start = now; % Record starting time
    disp(['  Simulation started: ', datestr(Start, 'dd-mmm-yyyy HH:MM:SS')])
    %OutputFile = 'longrun.mat';
    OutputFile = 'longrun_modifyp.mat';
    %OutputFile = strcat('trial',int2str(ii),'.mat');

    for t = 1:round(T)

        % Show progress at 1%, 10%, 20%, ... , 100%:
        if any(t == floor([1, 5:5:100]*T/100))
            Lap = now; 
            disp(['    ', num2str(round(100*t/T)), '%', ' Time elapsed: ', ...
                    datestr(Lap-Start, 'HH:MM:SS')])
            save(OutputFile);
        end

        % Filtered Poisson Step-like input:
        InputRateDt = (InitRate*(t<50/dt) + InputRate*(t>Tinit)*(t<Tinit+Tstim))*dt;  % (InitRate*(t<50/dt)
        New.SynO  = Old.SynO*SyndecayO + 1/dt*(1-SyndecayO)*(InputRateDt*ones(Next,1)>rand(Next,1));

        % Synaptic current fall assuming no spikes
        New.SynEE1 = Old.SynEE1*SyndecayEE1;
        New.SynEE2 = Old.SynEE2*SyndecayEE2;
        New.SynIE1 = Old.SynIE1*SyndecayIE1;
        New.SynIE2 = Old.SynIE2*SyndecayIE2;
        New.SynEI = Old.SynEI*SyndecayEI;
        New.SynII = Old.SynII*SyndecayII;

        % Second order Runge-Kutta algorithm for updating membrane potential
        Temp.Ex = 1/TauEx * (-Old.MemPotEx + JEE*ConnMatrix(1:Nex,1:Nex)*(pEE*Old.SynEE1+(1-pEE)*Old.SynEE2) + JEI*ConnMatrix(1:Nex,Nex+1:NAll)*Old.SynEI + JEO*InputConnMatrix(1:Nex,:)*Old.SynO);
        Temp.In = 1/TauIn * (-Old.MemPotIn + JIE*ConnMatrix(Nex+1:NAll,1:Nex)*(pIE*Old.SynIE1+(1-pIE)*Old.SynIE2) + JII*ConnMatrix(Nex+1:NAll,Nex+1:NAll)*Old.SynII  + JIO*InputConnMatrix(Nex+1:NAll,:)*Old.SynO);

        New.MemPotEx = Old.MemPotEx + dt/2 * (Temp.Ex ...
            +1/TauEx * (- (Old.MemPotEx+dt*Temp.Ex) + JEE*ConnMatrix(1:Nex,1:Nex)*(pEE*New.SynEE1+(1-pEE)*New.SynEE2) + JEI*ConnMatrix(1:Nex,Nex+1:NAll)*New.SynEI + JEO*InputConnMatrix(1:Nex,:)*New.SynO));
        New.MemPotIn = Old.MemPotIn + dt/2 * (Temp.In ...
            +1/TauIn * (- (Old.MemPotIn+dt*Temp.In) + JIE*ConnMatrix(Nex+1:NAll,1:Nex)*(pIE*New.SynIE1+(1-pIE)*New.SynIE2) + JII*ConnMatrix(Nex+1:NAll,Nex+1:NAll)*New.SynII + JIO*InputConnMatrix(Nex+1:NAll,:)*New.SynO)); 

        % setting membrane potential below threshold if neurons are in
        % refractorniness to prevent firing
        New.MemPotEx(Refr.Ex > 0) = 0; 
        New.MemPotIn(Refr.In > 0) = 0; 

        % Fire, reset MemPot, and update refractoriness:
        Spikes.Ex(:,t+1) = (New.MemPotEx >= ThreshEx);    
        Spikes.In(:,t+1) = (New.MemPotIn >= ThreshIn);    

        Index.Ex = logical(Spikes.Ex(:,t+1));
        Index.In = logical(Spikes.In(:,t+1));

        % Linear interporlation of spike times
        Tspikes.Ex(Index.Ex) = dt* (ThreshEx-Old.MemPotEx(Index.Ex)) ./ (New.MemPotEx(Index.Ex) - Old.MemPotEx(Index.Ex));
        Tspikes.In(Index.In) = dt* (ThreshIn-Old.MemPotIn(Index.In)) ./ (New.MemPotIn(Index.In) - Old.MemPotIn(Index.In));

        New.SynEE1(Index.Ex) = New.SynEE1(Index.Ex) + 1/TauSynEE1 * exp(-(dt-Tspikes.Ex(Index.Ex))/TauSynEE1);
        New.SynEE2(Index.Ex) = New.SynEE2(Index.Ex) + 1/TauSynEE2 * exp(-(dt-Tspikes.Ex(Index.Ex))/TauSynEE2);
        New.SynIE1(Index.Ex) = New.SynIE1(Index.Ex) + 1/TauSynIE1 * exp(-(dt-Tspikes.Ex(Index.Ex))/TauSynIE1);
        New.SynIE2(Index.Ex) = New.SynIE2(Index.Ex) + 1/TauSynIE2 * exp(-(dt-Tspikes.Ex(Index.Ex))/TauSynIE2);
        New.SynEI(Index.In) = New.SynEI(Index.In) + 1/TauSynEI * exp(-(dt-Tspikes.In(Index.In))/TauSynEI);
        New.SynII(Index.In) = New.SynII(Index.In) + 1/TauSynII * exp(-(dt-Tspikes.In(Index.In))/TauSynII);

        Refr.Ex = Refr.Ex - 1; Refr.Ex(Refr.Ex<0) = 0;          
        Refr.Ex(Index.Ex) = AbsRefractory;                      

        Refr.In = Refr.In - 1; Refr.In(Refr.In<0) = 0;          
        Refr.In(Index.In) = AbsRefractory;                      

        % Compute new membrane potential with first-order approximation of spike times 
        Temp.MemPotEx(Index.Ex) = ResetEx + (New.MemPotEx(Index.Ex)-ThreshEx) .* (1 + dt/TauEx * (Old.MemPotEx(Index.Ex)-ResetEx) ./ (New.MemPotEx(Index.Ex) - Old.MemPotEx(Index.Ex)));
        Temp.MemPotIn(Index.In) = ResetIn + (New.MemPotIn(Index.In)-ThreshIn) .* (1 + dt/TauIn * (Old.MemPotIn(Index.In)-ResetIn) ./ (New.MemPotIn(Index.In) - Old.MemPotIn(Index.In)));

        New.MemPotEx(logical(HistSpikes.Ex(:,end))) = HistMemPot.Ex(logical(HistSpikes.Ex(:,end)),end);
        New.MemPotIn(logical(HistSpikes.In(:,end))) = HistMemPot.In(logical(HistSpikes.In(:,end)),end);

        % Keep new membrane potential for refractory period
        HistSpikes.Ex = circshift(HistSpikes.Ex,[0 1]);
        HistSpikes.In = circshift(HistSpikes.In,[0 1]);

        HistMemPot.Ex = circshift(HistMemPot.Ex,[0 1]);
        HistMemPot.In = circshift(HistMemPot.In,[0 1]);

        HistSpikes.Ex(:,1) =0; HistSpikes.In(:,1) = 0;
        HistSpikes.Ex(Index.Ex,1) = 1;
        HistSpikes.In(Index.In,1) = 1;

        HistMemPot.Ex(Index.Ex,1) = Temp.MemPotEx(Index.Ex);
        HistMemPot.In(Index.In,1) = Temp.MemPotIn(Index.In);

        % Keep population firing statistics:
        Activity.Ex(t+1) = sum(Spikes.Ex(:,t+1)); 
        Activity.In(t+1) = sum(Spikes.In(:,t+1));

        % Update
        Old.SynO       = New.SynO;
        Old.SynEE1      = New.SynEE1;     
        Old.SynEE2      = New.SynEE2;     
        Old.SynEI      = New.SynEI;   
        Old.SynIE1      = New.SynIE1;   
        Old.SynIE2      = New.SynIE2;   
        Old.SynII      = New.SynII;
        Old.MemPotEx   = New.MemPotEx;  
        Old.MemPotIn   = New.MemPotIn;  
    
        if((t>Tinit+1000/dt)&&(t<Tinit+1500/dt)&&(mod((t-Tinit),round(1/dt))== 0))
        %if((t>500/dt)&&(t<Tinit)&&(mod((t),round(1/dt))== 0))    
                Stor.t = floor((t - (Tinit+1000/dt))*dt);
                %Stor.t = floor((t - (500/dt))*dt);
                Stor.MemPotEx(:,Stor.t) = New.MemPotEx; 
                Stor.MemPotIn(:,Stor.t) = New.MemPotIn; 
                Stor.SynEE1(:,Stor.t) = New.SynEE1; 
                Stor.SynEE2(:,Stor.t) = New.SynEE2; 
                Stor.SynEI(:,Stor.t) = New.SynEI; 
                Stor.SynIE1(:,Stor.t) = New.SynIE1; 
                Stor.SynIE2(:,Stor.t) = New.SynIE2; 
                Stor.SynII(:,Stor.t) = New.SynII; 
        end
    end  
end

%% Figures
LineWidth = 1;
FontSize = 12;

% Time course of average activity of excitatory neurons 
figure(1)
plot((0:T)*dt,smooth(Activity.Ex/Nex/dt*1000,1/dt),'k','Color',[0.5 0.5 0.5]);
hold on
plot((0:T)*dt,smooth(Activity.Ex/Nex/dt*1000,10/dt),'k','LineWidth',LineWidth);
hold off
xlim([Tinit*dt-200 T*dt-300]);ylim([0 70]);
set(gca,'FontSize',FontSize);
title('Time course of activity of excitatory population')
xlabel('Time (ms)');ylabel('Firing Rate (Hz)');
set(gca,'Xtick',Tinit*dt:1000:T*dt)
set(gca,'XTickLabel',0:1:4)
set(gca,'Ytick',0:20:100)
saveas(gcf,'figure1.png')

% Synaptic input during the delay period
figure(2)
plot(mean(JEE*ConnMatrix(1:Nex,1:Nex)*(pEE*Stor.SynEE1(:,1:end-1)+(1-pEE)*Stor.SynEE2(:,1:end-1)),1),'b','LineWidth',LineWidth)
hold on
plot(-JEI*mean(ConnMatrix(1:Nex,Nex+1:NAll)*Stor.SynEI(:,1:end-1),1),'r','LineWidth',LineWidth)
hold off
ylim([0 20])
set(gca,'FontSize',FontSize);
title('Time course of synaptic inputs onto excitatory population')
xlabel('Time (ms)');ylabel('Synaptic input');
legend('Exc','Inh')
set(gca,'Xtick',0:100:500)
set(gca,'XTickLabel',1:0.1:1.5) 
saveas(gcf,'figure2.png')

% CV of excitatory neurons
figure(3)
CV(1:Nex) = 100;
for i = 1:Nex
    clear ISI 
    index = find(Spikes.Ex(i,Tinit+1500/dt:end)>0);
    ISI = diff(index)*dt;
    
    if(length(index)>5);
        CV(i) = std(ISI)/mean(ISI);
    end
end

index_CV = find(CV<100);
hist(CV(index_CV),0:0.1:5);
h = findobj(gca,'Type','patch');
set(h,'FaceColor','w','EdgeColor','k')
set(gca,'FontSize',FontSize);
xlabel('CV');ylabel('Count');
xlim([0 5])
saveas(gcf,'figure3.png')

% CV of excitatory neurons
figure(4)
CV(1:Nex) = 100;
for i = 1:Nex
    clear ISI 
    index = find(Spikes.Ex(i,round(500/dt):Tinit)>0);
    ISI = diff(index)*dt;
    
    if(length(index)>5);
        CV(i) = std(ISI)/mean(ISI);
    end
end

index_CV = find(CV<100);
hist(CV(index_CV),0:0.1:5);
h = findobj(gca,'Type','patch');
set(h,'FaceColor','w','EdgeColor','k')
set(gca,'FontSize',FontSize);
xlabel('CV');ylabel('Count');
xlim([0 5])
saveas(gcf,'figure4.png')