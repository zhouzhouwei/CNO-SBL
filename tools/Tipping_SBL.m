function [x, iter, L1] = Tipping_SBL(y, Phi, paras, InitVal)

if nargin < 4
    gamma_init  = 1e-3;
    lambda_init = 1e-3;
else
    gamma_init = InitVal.gamma_init ;
    lambda_init = InitVal.lambda_init ;
end

threshold1 = paras.threshold;
iters = paras.iters ;
nmd = paras.normalized ;
delta = paras.delta;
a0 = paras.a0 ;
b0 = paras.b0 ;
c0 = paras.c0 ;
d0 = paras.d0 ;

[M,N] = size(Phi) ;
Gamma = gamma_init*eye(N) ;
Hygamma = diag(Gamma);
lambda = lambda_init ;
lambdas = zeros(iters,1);
errs = zeros(iters,1);
xhat = zeros(N,1) ;

% normlized data
if nmd ==1
    y_max = max(abs(y)) ;
    y = y/ y_max ;
    Phi_norm = vecnorm(Phi) ;
    Phi = Phi ./ vecnorm(Phi) ;
end
 
% main loop 
for iter = 1:iters
    xhat_old = xhat;
    %if Phi is over-determined using Woodbury identity to calculate Sigma
    if M > N
       Sigma = inv( diag(1./Hygamma)+Phi'*Phi/lambda ) ;
    else
       Sigma = Gamma - Gamma*Phi'*((lambda*eye(M)+Phi*Gamma*Phi')\Phi)*Gamma; 
    end
    xhat = 1/lambda*Sigma*Phi'*y ;

    
    %update the hyperparameters 
    temp = sum(1-diag(Sigma)./Hygamma) ;
    Hygamma = (xhat.^2 + diag(Sigma) + 2*b0 ) ./ (1+2*a0) ;
    lambda = (norm(y-Phi*xhat,2)^2 + lambda*temp + 2*d0)/( M + 2*c0) ;
    lambdas(iter,1) = lambda ;
    Gamma = diag(Hygamma) ;

    % stopping criterion 
    errs(iter) = norm(xhat - xhat_old)/norm(xhat);
    if errs(iter) <= delta 
        fprintf(1,'EM-SBL Algorithm converged, # iterations : %d \n',iter);
        break;
    end   
end

L1 = obj_SBL(y, Phi, xhat, Hygamma, lambda, paras);
x = xhat ;
if nmd ==1
    x = x * y_max ./Phi_norm' ;
end

%prune the small terms in x
x(abs(x)./norm(x)<threshold1) = 0;

end

%% objective function
function L = obj_SBL(y,Phi,w,gamma,sigma,paras)
a0 = paras.a0 ;
b0 = paras.b0 ;
c1 = 2*paras.c0 - 2 ;
d0 = paras.d0 ;
s0 = paras.a;
[m,n] = size(Phi);
n1 = n+2-m-2*a0 ;
H = norm(y-Phi*w)^2 ;
L = sum(log(sigma+s0*gamma) + c1*log(gamma) + (2*d0 + w.^2)./gamma ) ....
    + (H + 2*b0)./sigma - n1*log(sigma) ;
end