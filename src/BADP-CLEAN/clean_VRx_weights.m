function [weights] = clean_VRx_weights(phi, Y, weights_lsqlin)
%compute the weights to approximate a state

quad_options=optimoptions('quadprog','display','off');
weights_lsqlin(std(phi)<0.1)=0; %ignore constant features


N=length(phi(:,1));
%distance between the states
dist=zeros(N,1);
for n=1:N
    dist(n)=norm((phi(n,:)-Y).*weights_lsqlin);
end

dist_max=min(dist);
h_kernel=dist_max/sqrt(-log(0.5)); %kernal function
kernel_dist=zeros(N,1);
while(sum(kernel_dist)<2*log(N))
    kernel_dist=exp(- (dist/h_kernel).^2 );
    h_kernel=h_kernel+1;
end

lb=-kernel_dist;
ub=kernel_dist;

H=phi*diag(weights_lsqlin)*phi';
H=(H+H')/2;

A=[ones(1,N);
    -ones(1,N)];

eps=0; %can be used to allow the parameters not to sum to 1
b=[1+eps;-1+eps];

weights=quadprog(H,-phi*diag(weights_lsqlin)*Y',A,b,[],[],lb,ub,[],quad_options);

end

