function [weights]= badp_weights( T)
%computes the weights for badp-w

if nargin==0
    T=5;
end
D=7;    %days in forecast
Season='Summer';

gamma=1; %discount factor for futres prices

% load(strcat('Data\beta_day_ahead_',Season,'.mat'));
load(fullfile('Data', strcat('beta_day_ahead_', Season, '.mat')));
% load(strcat('Data\beta_intraday_',Season,'.mat'));
load(fullfile('Data', strcat('beta_intraday_', Season, '.mat')));


beta_day_ahead(:,1:8)=[]; %remove const and DoW
beta_intraday(:,1:8)=[];

einfluss_day=zeros(T,24*D);
for t=1:T
    for i=1:24*D
        Pt_day=zeros(1,24*D);
        Pt_day(i)=1;
        Pt_intraday=zeros(1,96*D);
        Wt_day_mat=zeros(T,24);
        Wt_intraday_mat=zeros(T,96);
        for t_strich=t:T
            Wt_day=Pt_day*beta_day_ahead';
            Wt_intraday=[Wt_day,Pt_day,Pt_intraday]*beta_intraday';
            
            Pt_day=[Wt_day,Pt_day(1:end-24)];
            Pt_intraday=[Wt_intraday,Pt_intraday(1:end-96)];
            Wt_day_mat(t_strich,:)=Wt_day;
            Wt_intraday_mat(t_strich,:)=Wt_intraday;
            
            einfluss_day(t,i)=einfluss_day(t,i)+(4*sum(abs(Wt_day)) +sum(abs(Wt_intraday)))*gamma^(t_strich-t);
        end
    end
end

einfluss_intraday=zeros(T,24*4*D);
for t=1:T
    for i=1:24*4*D
        Pt_day=zeros(1,24*D);
        Pt_intraday=zeros(1,96*D);
        Pt_intraday(i)=1;
        Wt_day_mat=zeros(T,24);
        Wt_intraday_mat=zeros(T,96);
        for t_strich=t:T
            Wt_day=Pt_day*beta_day_ahead';
            Wt_intraday=[Wt_day,Pt_day,Pt_intraday]*beta_intraday';
            
            Pt_day=[Wt_day,Pt_day(1:end-24)];
            Pt_intraday=[Wt_intraday,Pt_intraday(1:end-96)];
            Wt_day_mat(t_strich,:)=Wt_day;
            Wt_intraday_mat(t_strich,:)=Wt_intraday;
            einfluss_intraday(t,i)=einfluss_intraday(t,i)+(4*sum(abs(Wt_day)) +sum(abs(Wt_intraday)))*gamma^(t_strich-t);
        end
    end
end

weights=[einfluss_day,einfluss_intraday];
%save('weights.mat','weights')
end
