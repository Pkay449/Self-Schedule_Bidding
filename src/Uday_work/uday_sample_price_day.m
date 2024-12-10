function [ mu_P, cov_P ] = sample_price_day( Pt_day,t,Season)
% Pt aktuelle Preise am Day Ahead markt (Zeilenvektor)
% t= aktuelle STufe ==>samples für Stufe t+1

% mu_P = Erwartungswerte der Preise
% cov_P =Kovarianzmatrix
n_arg=nargin;
if(n_arg==0)
    t=1;
    Season='Summer';
    D=7;
end


load(strcat('Data\beta_day_ahead_',Season,'.mat'));
load(strcat('Data\cov_day_',Season,'.mat'));
load(strcat('Data\DoW_',Season,'.mat'));

%Reihenfolge Beta
% intercept, Beta_MO ... Beta_SO, Beta_(t-1,1)...Beta_(t-1,24) ...Beta(t-7,24),

DOW=zeros(1,7);
DOW(1+mod(t+DoW_P0-1,7))=1;

Q=[1,DOW,Pt_day];

mu_P=Q*beta_day_ahead';
cov_P=cov_day;

end

