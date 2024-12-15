function [ sample_P_day_all, sample_P_intraday_all ] = badp_w_p1(N,M,T,Season,length_R,seed)
% Berechnet optimale Mengenentscheidung f�r Pumpspeicherkraftwerk
% Um 12 Uhr werden 24 Mengengebote f�r Day Ahead Markt abgegeben
% Um 15 Uhr werden 96 Mengengebote f�r Intraday Markt abgebenen
% Entschieden wird die ein/ausgehende energie

%profile on
if(nargin==0)
    length_R=5;
    N=50;
    T=3;
    M=10;
    seed=1;
    Season='Summer';
end

D = 7; % days in forecast
rng(seed)
Rmax=100;
intlinprog_options=optimoptions('intlinprog','displa','off');

weights_D_value=badp_weights(T);


%lineare Rampe f�r an/abschalten von Pumpe/Turbine
t_ramp_pump_up=2/60;
t_ramp_pump_down=2/60;
t_ramp_turbine_up=2/60;
t_ramp_turbine_down=2/60;


c_grid_fee=5/4; %entspricht 5�/MWh
Delta_ti=0.25; %Intraday Block sind 15min
Delta_td=1;

Q_mult=1.2;      % Strafkostenmultiplikator bei Rammpe
Q_fix=3;        %Fixe Strafkosten bei Rampe

Q_start_pump=15; %Fixkosten beim starten der pumpen
Q_start_turbine=15; %Fixkosten bei starten der Turbinen


beta_pump=0.9; %Effizient beim Einspeisen.
beta_turbine=0.9; %Effizienz beim Ausspeisen


x_max_pump=10;  %Maximale Leistung der Pumpen (Eingehende Leistung)
x_min_pump=5;  %Minimale Leistung der Pumpen (Eingehende Leistung) (Wenn gelierfert wird)

x_max_turbine=10; %Maximale Leistung der Turbinen (Ausgehende Leistung)
x_min_turbine=5; %Minimale Leistung der Turbinen (Ausgehende Leistung)



R_vec=linspace(0,Rmax,length_R); %Speicherstand diskretisieren
x_vec=[-x_max_turbine,0,x_max_pump];


c_pump_up=t_ramp_pump_up/2;
c_pump_down=t_ramp_pump_down/2;
c_turbine_up=t_ramp_turbine_up/2;
c_turbine_down=t_ramp_turbine_down/2;

% Load Data
% load(strcat('Data\P_day_',Season,'.mat'));
% load(strcat('Data\P_intraday_',Season,'.mat'));

load(fullfile('Data', strcat('P_day_', Season, '.mat')));
load(fullfile('Data', strcat('P_intraday_', Season, '.mat')));


%Sample Path
sample_P_day_all=zeros(N,T,24*D);
sample_P_intraday_all=zeros(N,T,4*24*D);

Wt_day_mat=zeros(N,T*24);
Wt_intra_mat=zeros(N,T*96);
for n=1:N
    P_day=P_day_0(1:24*D);
    P_intraday=P_intraday_0(1:96*D);
    for t=1:T
        sample_P_day_all(n,t,:)=P_day;
        sample_P_intraday_all(n,t,:)=P_intraday;
        [mu_day,cor_day]=sample_price_day(P_day,t,Season);
        Wt_day=mvnrnd(mu_day,cor_day);
        
        [mu_intraday,cor_intraday]=sample_price_intraday([Wt_day,P_day],P_intraday,t,Season);
        Wt_intraday=mvnrnd(mu_intraday,cor_intraday);
        
        P_day=[Wt_day,P_day(1:end-24)];
        P_intraday=[Wt_intraday,P_intraday(1:end-96)];
        
        Wt_day_mat(n,(t-1)*24+1:t*24)=Wt_day;
        Wt_intra_mat(n,(t-1)*96+1:t*96)=Wt_intraday;
        
    end
end

% Get the current script's directory
currentDir = fileparts(mfilename('fullpath'));
OutputDir = fullfile(currentDir, 'Outputs');
if ~isfolder(OutputDir)
    mkdir(OutputDir);
end
% save results sample_P_day_all and sample_P_intraday_all
save(fullfile(OutputDir, 'sample_P_day_all.mat'), 'sample_P_day_all');
save(fullfile(OutputDir, 'sample_P_intraday_all.mat'), 'sample_P_intraday_all');

end