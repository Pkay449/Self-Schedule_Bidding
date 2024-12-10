function [ EV ] = badp_w_debug(N,M,T,Season,length_R,seed)
    % Berechnet optimale Mengenentscheidung für Pumpspeicherkraftwerk
    % Um 12 Uhr werden 24 Mengengebote für Day Ahead Markt abgegeben
    % Um 15 Uhr werden 96 Mengengebote für Intraday Markt abgegebenen
    % Entschieden wird die ein/ausgehende Energie
    
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
    intlinprog_options=optimoptions('intlinprog','display','off');
    
    weights_D_value=badp_weights(T);
    
    %lineare Rampe für an/abschalten von Pumpe/Turbine
    t_ramp_pump_up=2/60;
    t_ramp_pump_down=2/60;
    t_ramp_turbine_up=2/60;
    t_ramp_turbine_down=2/60;
    
    c_grid_fee=5/4; % entspricht 5 €/MWh
    Delta_ti=0.25; % Intraday Block sind 15min
    Delta_td=1;
    
    Q_mult=1.2;      % Strafkostenmultiplikator bei Rampe
    Q_fix=3;         % Fixe Strafkosten bei Rampe
    
    Q_start_pump=15; % Fixkosten beim Starten der Pumpen
    Q_start_turbine=15; % Fixkosten bei Starten der Turbinen
    
    beta_pump=0.9; % Effizienz beim Einspeisen
    beta_turbine=0.9; % Effizienz beim Ausspeisen
    
    x_max_pump=10;  % Maximale Leistung der Pumpen (Eingehende Leistung)
    x_min_pump=5;   % Minimale Leistung der Pumpen (Eingehende Leistung)
    
    x_max_turbine=10; % Maximale Leistung der Turbinen (Ausgehende Leistung)
    x_min_turbine=5;  % Minimale Leistung der Turbinen (Ausgehende Leistung)
    
    R_vec=linspace(0,Rmax,length_R); % Speicherstand diskretisieren
    x_vec=[-x_max_turbine,0,x_max_pump];
    
    c_pump_up=t_ramp_pump_up/2;
    c_pump_down=t_ramp_pump_down/2;
    c_turbine_up=t_ramp_turbine_up/2;
    c_turbine_down=t_ramp_turbine_down/2;
    
    % Load Data
    load(fullfile('Data', strcat('P_day_', Season, '.mat')));
    load(fullfile('Data', strcat('P_intraday_', Season, '.mat')));
    
    % Sample Path
    sample_P_day_all=zeros(N,T,24*D);
    sample_P_intraday_all=zeros(N,T,4*24*D);
    
    Wt_day_mat=zeros(N,T*24);
    Wt_intra_mat=zeros(N,T*96);
    n = 1;
    P_day=P_day_0(1:24*D);
    P_intraday=P_intraday_0(1:96*D);
    t = 1;
    sample_P_day_all(n,t,:)=P_day;
    sample_P_intraday_all(n,t,:)=P_intraday;
    
    % Save P_day and P_intraday
    [mu_day,cor_day]=sample_price_day(P_day,t,Season);
    Wt_day=mvnrnd(mu_day,cor_day);
    
    % Save debugging inputs for sample_price_day
    save('debug_sample_price_day.mat', 'P_day', 't', 'Season', 'mu_day', 'cor_day');
    
    [mu_intraday,cor_intraday]=sample_price_intraday([Wt_day,P_day],P_intraday,t,Season);
    Wt_intraday=mvnrnd(mu_intraday,cor_intraday);
    
    % Save debugging inputs for sample_price_intraday
    save('debug_sample_price_intraday.mat', 'Wt_day', 'P_day', 'P_intraday', 't', 'Season', 'mu_intraday', 'cor_intraday');
    
    end
    