function [ EV ] = badp_w(N,M,T,Season,length_R,seed)
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
    seed=2;
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


P_day_state=sample_P_day_all;
P_intra_state=sample_P_intraday_all;

% Lookup Table
Vt=zeros(length_R,3,N,T+1); 

for t=T:-1:1 %Backwards step
    P_day_sample=reshape(P_day_state(:,t,:),N,D*24);
    P_intraday_sample=reshape(P_intra_state(:,t,:),N,D*24*4);
    
    if t<T
        P_day_sample_next=reshape(P_day_state(:,t+1,:),N,D*24);
        P_intraday_sample_next=reshape(P_intra_state(:,t+1,:),N,D*24*4);
    end
    
    for n=1:N
        P_day=P_day_sample(n,:);
        P_intraday=P_intraday_sample(n,:);
        
        mu_day=sample_price_day(P_day,t,Season);
        mu_intraday=sample_price_intraday([mu_day,P_day],P_intraday,t,Season);
        
        
        P_day_next=[mu_day,P_day(1:end-24)];
        P_intraday_next=[mu_intraday,P_intraday(1:end-96)];
        
        lk=2;
        VR_abc_neg=zeros(lk-1,3);
        VR_abc_pos=zeros(lk-1,3);
         
        if t<T
            phi=[P_day_sample_next,P_intraday_sample_next];
            Y=[P_day_next,P_intraday_next];
            weights = VRx_weights(phi, Y, weights_D_value(t+1,:) );
            
            VRx=zeros(length_R,3);
            for i=1:length_R
                for j=1:3
                    VRx(i,j)=reshape(Vt(i,j,:,t+1),1,N)*weights;
                end
            end
            
            
            k=convhull([R_vec',VRx(:,2)]); %Konvexe H�lle f�r x=0
            k(1)=[]; %erstes element nicht n�tig;
            
            lk=length(k);
            %a Achsenabschnitt, b Steigung R, c Steigung x
            VR_abc_neg=zeros(lk-1,3);
            VR_abc_pos=zeros(lk-1,3);
            VR=VRx(k,:); %x=0
            R_k=R_vec(k);
            
            for i=2:lk
                VR_abc_neg(i-1,2)=(VR(i,2)-VR(i-1,2))/(R_k(i)-R_k(i-1)); %Steigung R
                VR_abc_neg(i-1,1)=VR(i,2)-VR_abc_neg(i-1,2)*R_k(i); %Achsenabschnitt
                VR_abc_neg(i-1,3)=-(VR(i-1,2)-VR(i-1,1))/(x_vec(2)-x_vec(1)); %Setigung x
            end
            
            for i=2:lk
                VR_abc_pos(i-1,2)=(VR(i,2)-VR(i-1,2))/(R_k(i)-R_k(i-1)); %Steigung R
                VR_abc_pos(i-1,1)=VR(i,2)-VR_abc_pos(i-1,2)*R_k(i); %Achsenabschnitt
                VR_abc_pos(i-1,3)=(VR(i-1,2)-VR(i-1,3))/(x_vec(2)-x_vec(3)); %Setigung x
            end
            
        end
        
        for iR=1:length_R %�ber alle Speicherst�nde
            R=R_vec(iR);
            for ix=1:length(x_vec) %�ber sign(x0)
                x0=x_vec(ix);
                f=zeros(1,96*12+24+1);
                %Wertfunktion
                f(end)=1;
                
                %Einstufige Zielfunktion
                f(96+1:96*2)=f(96+1:96*2)-Delta_ti*(mu_intraday);
                f(end-24:end-1)=-Delta_td*(mu_day);
                
                q_pump_up=(abs(mu_intraday)/Q_mult-Q_fix)*t_ramp_pump_up/2;
                q_pump_down=(abs(mu_intraday)*Q_mult+Q_fix)*t_ramp_pump_down/2;
                q_turbine_up=(abs(mu_intraday)*Q_mult+Q_fix)*t_ramp_turbine_up/2;
                q_turbine_down=(abs(mu_intraday)/Q_mult-Q_fix)*t_ramp_turbine_down/2;
                
                %grid fee
                f(96*2+1:96*3)=f(96*2+1:96*3)-c_grid_fee;
                %delta_pump_up
                f(96*4+1:96*5)=f(96*4+1:96*5)+q_pump_up;
                %delta_pump_down
                f(96*5+1:96*6)=f(96*5+1:96*6)-q_pump_down;
                %delta_turbine_up
                f(96*6+1:96*7)=f(96*6+1:96*7)-q_turbine_up;
                %delta_turbine_down
                f(96*7+1:96*8)=f(96*7+1:96*8)+q_turbine_down;
                
                %z^pump
                f(96*10+1:96*11)=f(96*10+1:96*11)-Q_start_pump;
                %z^turbine
                f(96*11+1:96*12)=f(96*11+1:96*12)-Q_start_turbine;
                
                
                %Lineare Gleichungsbedingungen
                %R_hq
                A1=[-eye(96)+diag(ones(95,1),-1),zeros(96),Delta_ti*beta_pump*eye(96),-Delta_ti/beta_turbine*eye(96),-beta_pump*c_pump_up*eye(96),beta_pump*c_pump_down*eye(96),c_turbine_up/beta_turbine*eye(96),-c_turbine_down/beta_turbine*eye(96),zeros(96,96*4+24),zeros(96,1)];
                b1=zeros(96,1);
                b1(1)=-R;
                
                %Aufteilung in x_pump und x_turbine
                Axh=zeros(96,24);
                for h=1:24
                    Axh((h-1)*4+1:h*4,h)=-1;
                end
                
                A2=[zeros(96),-eye(96),eye(96),-eye(96),zeros(96,96*8),Axh,zeros(96,1)];
                b2=zeros(96,1);
               
                %Aufteilung in x_pump_up, x_pump_turbine
                A3=[zeros(96,96*2),eye(96)-diag(ones(95,1),-1),zeros(96),-eye(96),eye(96),zeros(96,96*6+24),zeros(96,1)];
                b3=zeros(96,1);
                b3(1)=max(x0,0);
                
                %Aufteilung in x_turbine_up x_turbine_down
                A4=[zeros(96,96*3),eye(96)-diag(ones(95,1),-1),zeros(96,96*2),-eye(96),eye(96),zeros(96,96*4+24),zeros(96,1)];
                b4=zeros(96,1);
                b4(1)=max(-x0,0);
                       
                Aeq=[A1;A2;A3;A4];
                beq=[b1;b2;b3;b4];
                
                %Nur Pumpen wenn Pumpe an
                A1=[zeros(96,96*2),-eye(96),zeros(96,96*5),x_min_pump*eye(96),zeros(96,96*3+24),zeros(96,1);
                    zeros(96,96*2),eye(96),zeros(96,96*5),-x_max_pump*eye(96),zeros(96,96*3+24),zeros(96,1);
                    zeros(96,96*3),-eye(96),zeros(96,96*5),x_min_turbine*eye(96),zeros(96,96*2+24),zeros(96,1);
                    zeros(96,96*3),eye(96),zeros(96,96*5),-x_max_turbine*eye(96),zeros(96,96*2+24),zeros(96,1)];
                
                b1=zeros(96*4,1);
                
                %Wann wird Pumpe/turbine angestellt
                A2=[zeros(96,96*8),eye(96)-diag(ones(1,95),-1),zeros(96,96),-eye(96),zeros(96,96+24),zeros(96,1)];
                b2=zeros(96,1);
                b2(1)=x0>0;
                
                A3=[zeros(96,96*9),eye(96)-diag(ones(1,95),-1),zeros(96,96),-eye(96),zeros(96,24),zeros(96,1)];
                b3=zeros(96,1);
                b3(1)=x0<0;
                
                %nur pumpe oder turbine anstellen
                
                A4=[zeros(96,96*8),eye(96),eye(96),zeros(96,2*96+24),zeros(96,1)];
                b4=ones(96,1);
                
                %restriktionen f�r Wertunktion
                AV_neg=zeros(lk-1,12*96+24+1);
                AV_neg(:,end)=1;
                AV_neg(:,96)=-VR_abc_neg(:,2); %Speicher
                AV_neg(:,4*96)=-VR_abc_neg(:,3); %turbine
                bV_neg=VR_abc_neg(:,1);
                
                %restriktionen f�r Wertunktion
                AV_pos=zeros(lk-1,12*96+24+1);
                AV_pos(:,end)=1;
                AV_pos(:,96)=-VR_abc_neg(:,2); %Speicher
                AV_pos(:,3*96)=-VR_abc_neg(:,3); %pumpe
                bV_pos=VR_abc_pos(:,1);
                
                A=[A1;A2;A3;A4;AV_neg;AV_pos];
                b=[b1;b2;b3;b4;bV_neg;bV_pos];
                
                lb=[zeros(1,96),-inf*ones(1,96),zeros(1,96*10),-x_max_turbine*ones(1,24),-inf];
                ub=[ones(1,96)*Rmax,inf*ones(1,96*7),ones(1,96*4),x_max_pump*ones(1,24),inf];
                
                intcon=8*96+1:96*10;
                
                [~,fval,~,~]=intlinprog(-f,intcon,A,b,Aeq,beq,lb,ub,intlinprog_options);
                %Definitionsbereich der Variablen
                
                Vt(iR,ix,n,t)=-fval;               
            end
        end     
    end 
end

rng(seed+1)

% Simulation with price process
% Sample path

sample_P_day_all=zeros(N,T,24*D);
sample_P_intraday_all=zeros(N,T,4*24*D);

Wt_day_mat=zeros(M,T*24);
Wt_intra_mat=zeros(M,T*96);
for n=1:M
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

R_0=0;
x0_0=0;

V=zeros(M,1);

R_path=zeros(M,96*T);
x_intraday_path=zeros(M,96*T);
P_day_path=zeros(M,96*T);
P_intraday_path=zeros(M,96*T);

x_pump_path=zeros(M,96*T);
x_turbine_path=zeros(M,96*T);
y_pump_path=zeros(M,96*T);
y_turbine_path=zeros(M,96*T);
z_pump_path=zeros(M,96*T);
z_turbine_path=zeros(M,96*T);

for m=1:M
    
    R=R_0;
    x0=x0_0;
    P_day=P_day_0(1:24*D);
    P_intraday=P_intraday_0((1:96*D));
    
    P_day_sim=P_day_0(1:24*D);
    P_intraday_sim=P_intraday_0((1:96*D));
    
    C=0;
    for t=1:T
        [mu_day,~]=sample_price_day(P_day_sim,t,Season);
        [mu_intraday,~]=sample_price_intraday([mu_day,P_day_sim],P_intraday_sim,t,Season);
        
        P_day_next=[mu_day,P_day(1:end-24)];
        P_intraday_next=[mu_intraday,P_intraday(1:end-96)];
        

        lk=2;
        VR_abc_neg=zeros(lk-1,3);
        VR_abc_pos=zeros(lk-1,3);
        
        if (t<T &&any(Vt(:)~=0 ))
            P_day_sample_next=reshape(P_day_state(:,t+1,:),N,D*24);
            P_intraday_sample_next=reshape(P_intra_state(:,t+1,:),N,D*24*4);
            
            phi=[P_day_sample_next,P_intraday_sample_next];
            Y=[P_day_next,P_intraday_next];
            
            weights = VRx_weights(phi, Y, weights_D_value(t+1,:) );
            
            
            VRx=zeros(length_R,3);
            for i=1:length_R
                for j=1:3
                    VRx(i,j)=reshape(Vt(i,j,:,t+1),1,N)*weights;
                end
            end
            k=convhull([R_vec',VRx(:,2)]); %Konvexe H�lle f�r x=0
            k(1)=[]; %erstes element nicht n�tig;
            
            lk=length(k);
            %a Achsenabschnitt, b Steigung R, c Steigung x
            VR_abc_neg=zeros(lk-1,3);
            VR_abc_pos=zeros(lk-1,3);
            VR=VRx(k,:); %x=0
            R_k=R_vec(k);
            
            for i=2:lk
                VR_abc_neg(i-1,2)=(VR(i,2)-VR(i-1,2))/(R_k(i)-R_k(i-1)); %Steigung R
                VR_abc_neg(i-1,1)=VR(i,2)-VR_abc_neg(i-1,2)*R_k(i); %Achsenabschnitt
                VR_abc_neg(i-1,3)=-(VR(i-1,2)-VR(i-1,1))/(x_vec(2)-x_vec(1)); %Setigung x
            end
            
            for i=2:lk
                VR_abc_pos(i-1,2)=(VR(i,2)-VR(i-1,2))/(R_k(i)-R_k(i-1)); %Steigung R
                VR_abc_pos(i-1,1)=VR(i,2)-VR_abc_pos(i-1,2)*R_k(i); %Achsenabschnitt
                VR_abc_pos(i-1,3)=(VR(i-1,2)-VR(i-1,3))/(x_vec(2)-x_vec(3)); %Setigung x
            end
        end
        
        f=zeros(1,96*12+24+1);
        %Wertfunktion
        f(end)=1;
        
        
        %Einstufige Zielfunktion
        f(96+1:96*2)=f(96+1:96*2)-Delta_ti*(mu_intraday);
        f(end-24:end-1)=-Delta_td*(mu_day);
        
        q_pump_up=(abs(mu_intraday)/Q_mult-Q_fix)*t_ramp_pump_up/2;
        q_pump_down=(abs(mu_intraday)*Q_mult+Q_fix)*t_ramp_pump_down/2;
        q_turbine_up=(abs(mu_intraday)*Q_mult+Q_fix)*t_ramp_turbine_up/2;
        q_turbine_down=(abs(mu_intraday)/Q_mult-Q_fix)*t_ramp_turbine_down/2;
        
        %grid fee
        f(96*2+1:96*3)=f(96*2+1:96*3)-c_grid_fee;
        %delta_pump_up
        f(96*4+1:96*5)=f(96*4+1:96*5)+q_pump_up;
        %delta_pump_down
        f(96*5+1:96*6)=f(96*5+1:96*6)-q_pump_down;
        %delta_turbine_up
        f(96*6+1:96*7)=f(96*6+1:96*7)-q_turbine_up;
        %delta_turbine_down
        f(96*7+1:96*8)=f(96*7+1:96*8)+q_turbine_down;
        
        %z^pump
        f(96*10+1:96*11)=f(96*10+1:96*11)-Q_start_pump;
        %z^turbine
        f(96*11+1:96*12)=f(96*11+1:96*12)-Q_start_turbine;
        
        
        %Lineare Gleichungsbedingungen
        %R_hq
        A1=[-eye(96)+diag(ones(95,1),-1),zeros(96),Delta_ti*beta_pump*eye(96),-Delta_ti/beta_turbine*eye(96),-beta_pump*c_pump_up*eye(96),beta_pump*c_pump_down*eye(96),c_turbine_up/beta_turbine*eye(96),-c_turbine_down/beta_turbine*eye(96),zeros(96,96*4+24),zeros(96,1)];
        b1=zeros(96,1);
        b1(1)=-R;
        
        %Aufteilung in x_pump und x_turbine
        Axh=zeros(96,24);
        for h=1:24
            Axh((h-1)*4+1:h*4,h)=-1;
        end
        
        A2=[zeros(96),-eye(96),eye(96),-eye(96),zeros(96,96*8),Axh,zeros(96,1)];
        b2=zeros(96,1);

        %Aufteilung in x_pump_up, x_pump_turbine
        A3=[zeros(96,96*2),eye(96)-diag(ones(95,1),-1),zeros(96),-eye(96),eye(96),zeros(96,96*6+24),zeros(96,1)];
        b3=zeros(96,1);
        b3(1)=max(x0,0);
        
        %Aufteilung in x_turbine_up x_turbine_down
        A4=[zeros(96,96*3),eye(96)-diag(ones(95,1),-1),zeros(96,96*2),-eye(96),eye(96),zeros(96,96*4+24),zeros(96,1)];
        b4=zeros(96,1);
        b4(1)=max(-x0,0);
              
        
        Aeq=[A1;A2;A3;A4];
        beq=[b1;b2;b3;b4];
        
        %Nur Pumpen wenn Pumpe an
        A1=[zeros(96,96*2),-eye(96),zeros(96,96*5),x_min_pump*eye(96),zeros(96,96*3+24),zeros(96,1);
            zeros(96,96*2),eye(96),zeros(96,96*5),-x_max_pump*eye(96),zeros(96,96*3+24),zeros(96,1);
            zeros(96,96*3),-eye(96),zeros(96,96*5),x_min_turbine*eye(96),zeros(96,96*2+24),zeros(96,1);
            zeros(96,96*3),eye(96),zeros(96,96*5),-x_max_turbine*eye(96),zeros(96,96*2+24),zeros(96,1)];
        
        b1=zeros(96*4,1);
        
        %Wann wird Pumpe/turbine angestellt
        A2=[zeros(96,96*8),eye(96)-diag(ones(1,95),-1),zeros(96,96),-eye(96),zeros(96,96+24),zeros(96,1)];
        b2=zeros(96,1);
        b2(1)=x0>0;
        
        A3=[zeros(96,96*9),eye(96)-diag(ones(1,95),-1),zeros(96,96),-eye(96),zeros(96,24),zeros(96,1)];
        b3=zeros(96,1);
        b3(1)=x0<0;
        
        %nur pumpe oder turbine anstellen
        
        A4=[zeros(96,96*8),eye(96),eye(96),zeros(96,2*96+24),zeros(96,1)];
        b4=ones(96,1);
        
        %restriktionen f�r Wertunktion
        AV_neg=zeros(lk-1,12*96+24+1);
        AV_neg(:,end)=1;
        AV_neg(:,96)=-VR_abc_neg(:,2); %Speicher
        AV_neg(:,4*96)=-VR_abc_neg(:,3); %turbine
        bV_neg=VR_abc_neg(:,1);
        
        %restriktionen f�r Wertunktion
        AV_pos=zeros(lk-1,12*96+24+1);
        AV_pos(:,end)=1;
        AV_pos(:,96)=-VR_abc_neg(:,2); %Speicher
        AV_pos(:,3*96)=-VR_abc_neg(:,3); %pumpe
        bV_pos=VR_abc_pos(:,1);
        
        A=[A1;A2;A3;A4;AV_neg;AV_pos];
        b=[b1;b2;b3;b4;bV_neg;bV_pos];
        
        lb=[zeros(1,96),-inf*ones(1,96),zeros(1,96*10),-x_max_turbine*ones(1,24),-inf];
        ub=[ones(1,96)*Rmax,inf*ones(1,96*7),ones(1,96*4),x_max_pump*ones(1,24),inf];
        
        intcon=8*96+1:96*10;
        
        [x_opt,~,~,~]=intlinprog(-f,intcon,A,b,Aeq,beq,lb,ub,intlinprog_options);
        
        xday_opt=x_opt(end-24:end-1);
        
        
        Wt_day=Wt_day_mat(m,(t-1)*24+1:t*24);
        day_path=repmat(Wt_day,4,1);
        P_day_path(m,(t-1)*96+1:t*96)=day_path(:)';
        
        [mu_intraday,~]=sample_price_intraday([Wt_day,P_day_sim],P_intraday_sim,t,Season);
        
        
        P_day_next=[Wt_day,P_day(1:end-24)];
        P_intraday_next=[mu_intraday,P_intraday(1:end-96)];
        
        lk=2;
        %a Achsenabschnitt, b Steigung R, c Steigung x
        VR_abc_neg=zeros(lk-1,3);
        VR_abc_pos=zeros(lk-1,3);
        if (t<T &&any(Vt(:)~=0 ))
            phi=[P_day_sample_next,P_intraday_sample_next];
            Y=[P_day_next,P_intraday_next];
            
            weights = VRx_weights(phi, Y, weights_D_value(t+1,:) );
            
            VRx=zeros(length_R,3);
            for i=1:length_R
                for j=1:3
                    VRx(i,j)=reshape(Vt(i,j,:,t+1),1,N)*weights;
                    
                end
            end
            
            lk=length(k);
            %a Achsenabschnitt, b Steigung R, c Steigung x
            VR_abc_neg=zeros(lk-1,3);
            VR_abc_pos=zeros(lk-1,3);
            VR=VRx(k,:); %x=0
            R_k=R_vec(k);
            
            for i=2:lk
                VR_abc_neg(i-1,2)=(VR(i,2)-VR(i-1,2))/(R_k(i)-R_k(i-1)); %Steigung R
                VR_abc_neg(i-1,1)=VR(i,2)-VR_abc_neg(i-1,2)*R_k(i); %Achsenabschnitt
                VR_abc_neg(i-1,3)=-(VR(i-1,2)-VR(i-1,1))/(x_vec(2)-x_vec(1)); %Setigung x
            end
            
            for i=2:lk
                VR_abc_pos(i-1,2)=(VR(i,2)-VR(i-1,2))/(R_k(i)-R_k(i-1)); %Steigung R
                VR_abc_pos(i-1,1)=VR(i,2)-VR_abc_pos(i-1,2)*R_k(i); %Achsenabschnitt
                VR_abc_pos(i-1,3)=(VR(i-1,2)-VR(i-1,3))/(x_vec(2)-x_vec(3)); %Setigung x
            end
            
        end
        
        f=zeros(1,96*12+24+1);
        %Wertfunktion
        f(end)=1;
        
        
        %Einstufige Zielfunktion
        f(96+1:96*2)=f(96+1:96*2)-Delta_ti*(mu_intraday);
        f(end-24:end-1)=-Delta_td*(mu_day);
        
        q_pump_up=(abs(mu_intraday)/Q_mult-Q_fix)*t_ramp_pump_up/2;
        q_pump_down=(abs(mu_intraday)*Q_mult+Q_fix)*t_ramp_pump_down/2;
        q_turbine_up=(abs(mu_intraday)*Q_mult+Q_fix)*t_ramp_turbine_up/2;
        q_turbine_down=(abs(mu_intraday)/Q_mult-Q_fix)*t_ramp_turbine_down/2;
        
        %grid fee
        f(96*2+1:96*3)=f(96*2+1:96*3)-c_grid_fee;
        %delta_pump_up
        f(96*4+1:96*5)=f(96*4+1:96*5)+q_pump_up;
        %delta_pump_down
        f(96*5+1:96*6)=f(96*5+1:96*6)-q_pump_down;
        %delta_turbine_up
        f(96*6+1:96*7)=f(96*6+1:96*7)-q_turbine_up;
        %delta_turbine_down
        f(96*7+1:96*8)=f(96*7+1:96*8)+q_turbine_down;
        
        %z^pump
        f(96*10+1:96*11)=f(96*10+1:96*11)-Q_start_pump;
        %z^turbine
        f(96*11+1:96*12)=f(96*11+1:96*12)-Q_start_turbine;
        
        %Lineare Gleichungsbedingungen
        %R_hq
        A1=[-eye(96)+diag(ones(95,1),-1),zeros(96),Delta_ti*beta_pump*eye(96),-Delta_ti/beta_turbine*eye(96),-beta_pump*c_pump_up*eye(96),beta_pump*c_pump_down*eye(96),c_turbine_up/beta_turbine*eye(96),-c_turbine_down/beta_turbine*eye(96),zeros(96,96*4+24),zeros(96,1)];
        b1=zeros(96,1);
        b1(1)=-R;
        
        %Aufteilung in x_pump und x_turbine
        Axh=zeros(96,24);
        for h=1:24
            Axh((h-1)*4+1:h*4,h)=-1;
        end
        
        A2=[zeros(96),-eye(96),eye(96),-eye(96),zeros(96,96*8),Axh,zeros(96,1)];
        b2=zeros(96,1);
        %         for i=1:24
        %             b2((i-1)*4+1:4*i)=x_day(i);
        %         end
        
        
        
        %Aufteilung in x_pump_up, x_pump_turbine
        A3=[zeros(96,96*2),eye(96)-diag(ones(95,1),-1),zeros(96),-eye(96),eye(96),zeros(96,96*6+24),zeros(96,1)];
        b3=zeros(96,1);
        b3(1)=max(x0,0);
        
        %Aufteilung in x_turbine_up x_turbine_down
        A4=[zeros(96,96*3),eye(96)-diag(ones(95,1),-1),zeros(96,96*2),-eye(96),eye(96),zeros(96,96*4+24),zeros(96,1)];
        b4=zeros(96,1);
        b4(1)=max(-x0,0);
               
        Aeq=[A1;A2;A3;A4];
        beq=[b1;b2;b3;b4];
        
        %Nur Pumpen wenn Pumpe an
        A1=[zeros(96,96*2),-eye(96),zeros(96,96*5),x_min_pump*eye(96),zeros(96,96*3+24),zeros(96,1);
            zeros(96,96*2),eye(96),zeros(96,96*5),-x_max_pump*eye(96),zeros(96,96*3+24),zeros(96,1);
            zeros(96,96*3),-eye(96),zeros(96,96*5),x_min_turbine*eye(96),zeros(96,96*2+24),zeros(96,1);
            zeros(96,96*3),eye(96),zeros(96,96*5),-x_max_turbine*eye(96),zeros(96,96*2+24),zeros(96,1)];
        
        b1=zeros(96*4,1);
        
        %Wann wird Pumpe/turbine angestellt
        A2=[zeros(96,96*8),eye(96)-diag(ones(1,95),-1),zeros(96,96),-eye(96),zeros(96,96+24),zeros(96,1)];
        b2=zeros(96,1);
        b2(1)=x0>0;
        
        A3=[zeros(96,96*9),eye(96)-diag(ones(1,95),-1),zeros(96,96),-eye(96),zeros(96,24),zeros(96,1)];
        b3=zeros(96,1);
        b3(1)=x0<0;
        
        %nur pumpe oder turbine anstellen
        
        A4=[zeros(96,96*8),eye(96),eye(96),zeros(96,2*96+24),zeros(96,1)];
        b4=ones(96,1);
        
        %restriktionen f�r Wertunktion
        AV_neg=zeros(lk-1,12*96+24+1);
        AV_neg(:,end)=1;
        AV_neg(:,96)=-VR_abc_neg(:,2); %Speicher
        AV_neg(:,4*96)=-VR_abc_neg(:,3); %turbine
        bV_neg=VR_abc_neg(:,1);
        
        %restriktionen f�r Wertunktion
        AV_pos=zeros(lk-1,12*96+24+1);
        AV_pos(:,end)=1;
        AV_pos(:,96)=-VR_abc_neg(:,2); %Speicher
        AV_pos(:,3*96)=-VR_abc_neg(:,3); %pumpe
        bV_pos=VR_abc_pos(:,1);
        
        A=[A1;A2;A3;A4;AV_neg;AV_pos];
        b=[b1;b2;b3;b4;bV_neg;bV_pos];
        
        lb=[zeros(1,96),-inf*ones(1,96),zeros(1,96*10),xday_opt',-inf];
        ub=[ones(1,96)*Rmax,inf*ones(1,96*7),ones(1,96*4),xday_opt',inf];
        
        intcon=8*96+1:96*10;
        
        [x_opt,~,~,~]=intlinprog(-f,intcon,A,b,Aeq,beq,lb,ub,intlinprog_options);
        

        R_opt=x_opt(1:96);
        xhq_opt=x_opt(1+96:2*96);
        
        Delta_pump_up=x_opt(4*96+1:5*96);
        Delta_pump_down=x_opt(5*96+1:6*96);
        Delta_turbine_up=x_opt(6*96+1:7*96);
        Delta_turbine_down=x_opt(7*96+1:8*96);
        
        x_pump=x_opt(2*96+1:3*96);
        x_turbine=x_opt(3*96+1:4*96);
        y_pump=x_opt(8*96+1:9*96);
        y_turbine=x_opt(9*96+1:10*96);
        z_pump=x_opt(10*96+1:11*96);
        z_turbine=x_opt(11*96+1:12*96);
        
        
        R_path(m,(t-1)*96+1:96*t)=R_opt';
        x_intraday_path(m,(t-1)*96+1:96*t)=xhq_opt';
        
        x_pump_path(m,(t-1)*96+1:96*t)=x_pump';
        x_turbine_path(m,(t-1)*96+1:96*t)=x_turbine';
        y_pump_path(m,(t-1)*96+1:96*t)=y_pump';
        y_turbine_path(m,(t-1)*96+1:96*t)=y_turbine';
        z_pump_path(m,(t-1)*96+1:96*t)=z_pump;
        z_turbine_path(m,(t-1)*96+1:96*t)=z_turbine';
        
        Wt_intraday=Wt_intra_mat(m,(t-1)*96+1:t*96);
        P_intraday_path(m,(t-1)*96+1:96*t)=Wt_intraday;
        
        q_pump_up=(abs(Wt_intraday)/Q_mult-Q_fix)*t_ramp_pump_up/2;
        q_pump_down=(abs(Wt_intraday)*Q_mult+Q_fix)*t_ramp_pump_down/2;
        q_turbine_up=(abs(Wt_intraday)*Q_mult+Q_fix)*t_ramp_turbine_up/2;
        q_turbine_down=(abs(Wt_intraday)/Q_mult-Q_fix)*t_ramp_turbine_down/2;
         
        R=R_opt(end);
        x0=x_pump(end)-x_turbine(end);
        P_day=[Wt_day,P_day(1:end-24)];
        P_intraday=[Wt_intraday,P_intraday(1:end-96)];
        P_day_sim=[Wt_day,P_day_sim(1:end-24)];
        P_intraday_sim=[Wt_intraday,P_intraday_sim(1:end-96)];
        
        C=C-Delta_td*Wt_day*xday_opt-sum(x_pump)*c_grid_fee...
            -Delta_ti*Wt_intraday*xhq_opt+...
            q_pump_up*Delta_pump_up-q_pump_down*Delta_pump_down-...
            q_turbine_up*Delta_turbine_up+q_turbine_down*Delta_turbine_down-...
            sum(z_pump)*Q_start_pump-sum(z_turbine)*Q_start_turbine;
           
    end
    
    V(m)=C;
end

EV=mean(V);
disp(EV)

end