% demo to show the call of ICA_EBM
clear
close all

alpha = 1:1:20;
N = length(alpha);
T = 2000;
s = rand(N,T);
for n = 1 : N
    s(n,:) = gamrnd(alpha(n),1,1,T);    % these are Gamma sources
end
A = randn(N,N); 
x = A*s;

W = ICA_EBM(x);

figure;
T = W*A;
for n=1:N
    T(n,:)=T(n,:)/(max(abs(T(n,:))));
end
T=kron(T,ones(50,50));
imshow(abs(T))
title('Confusion matrix')






