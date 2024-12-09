function [ mu_P, cov_P ] = sample_price_intraday( Pt_day,Pt_intraday,t,Season)
% Pt aktuelle Preise am Day Ahead markt inklusive von Heute (Zeilenvektor)
% Pt_intraday aktuelle Preise am Intraday Markt (Zeilenvektor)
% t= aktuelle Stufe ==>samples für Stufe t+1

%mu_P = Erwartungswerte der Preise
% cov_P =Kovarianzmatrix
n_arg=nargin;
if(n_arg==0)
    t=1;
    Season='Summer';
end

load(strcat('Data\beta_day_ahead_',Season,'.mat'));
load(strcat('Data\cov_day_',Season,'.mat'));
load(strcat('Data\beta_intraday_',Season,'.mat'));
load(strcat('Data\cov_intraday_',Season,'.mat'));
% intercept, Beta_MO ... Beta_SO,Beta_intraday, Beta_(t-0,1)...Beta_(t-0,24) ...Beta(t-7,24),
load(strcat('Data\DoW_',Season,'.mat'));

DOW=zeros(1,7);
DOW(1+mod(t+DoW_P0-1,7))=1;

Q=[1,DOW,Pt_intraday,Pt_day];

mu_P=Q*beta_intraday';
cov_P=cov_intraday;

end

