function [xhat, Lobj] =IFSBL(A,y,paras, InitVal)
% the code for paper tiled "Fast Inverse-Free Sparse Bayesian Learning
% via Relaxed Evidence Lower Bound Maximization"

[M, N] = size(A);
Max_iter = paras.iters ;
if nargin < 4
    alpha = ones(N,1);
    sigma2 = 1 ;
    mu = A'*y;
else
    alpha = 1/InitVal.gamma_init ;
    sigma2 = 1/InitVal.lambda_init ;
    mu = InitVal.beta_init ;
end
a = paras.a0 ;
b = paras.b0 ;
c = paras.c0 ;
d = paras.d0 ;
threshold1 = paras.threshold;
nmd = paras.normalized ;

% normlized data
if nmd ==1
    y_max = max(abs(y)) ;
    y = y/ y_max ;
    A_norm = vecnorm(A) ;
    A = A ./ vecnorm(A) ;
end

L = 2*paras.a ;  % Lipschitz constant
theta = mu;
ATY = A'*y;
ATA = A'*A;
Ath = A*theta;
for iter = 1 : Max_iter
    mu_old = mu;
    sigma = 1./(sigma2*L/2+alpha);  % Calculate the disgnoal entries of the covariance matrix
    mu = (sigma2*theta*L+2*sigma2*(ATY-ATA*theta)).*sigma/2;    % Update mu
    alpha = (0.5+a)./(0.5*(mu.^2+sigma)+b); % Update alpha
    Amu = A*mu;
    sigma2 = (c+0.5*M)/(d+0.5*((y-2*Amu+Ath)'*(y-Ath)+0.5*L*sum((mu-theta).^2)+0.5*L*sum(sigma)));% Update noise precision
    theta = mu; % Update theta
    Ath = Amu;
    if norm(mu-mu_old)/norm(mu)< paras.delta
        fprintf(1,'IF-SBL Algorithm converged, # iterations : %d \n',iter);
        break
    end
end

xhat = mu;
gamma = 1./alpha ;
lambda = 1/ sigma2 ;
Lobj = obj_SBL(y, A, xhat, gamma, lambda, paras);
if nmd == 1
    xhat = xhat * y_max ./ A_norm' ;
end

xhat(abs(xhat)/norm(xhat)<threshold1) = 0;

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
