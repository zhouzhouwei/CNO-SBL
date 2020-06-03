function [xhat, obj_val] = PNN_ode23s(y, Phi, paras, InitVal)
[~,n] = size(Phi) ;

if nargin <4
    %initilization for PNN
    w_init = randn(n,1)  ;
    gamma_init = ones(n,1)*rand(1) ;
    sigma_init = rand(1) ;
else
    w_init = InitVal.beta_init;
    gamma_init = ones(n,1)*InitVal.gamma_init ;
    sigma_init = InitVal.lambda_init ;
end

threshold = paras.threshold;
tspan =  paras.Tspan ;
epsilons = 1e-6;  % time constant 
nmd = paras.normalized ;
if nmd ==1 
    y_max = max(abs(y)) ;
    y = y/ y_max ;
    Phi_norm = vecnorm(Phi) ;
    Phi = Phi ./ vecnorm(Phi) ;
end

u0 = [w_init; gamma_init; sigma_init];


% neurodynamic model
[t,ut] = ode23s(@(t,xx) fun1(t, xx, epsilons, y, Phi, paras), tspan, u0) ;

% output equation via projection
rnn_i = ut(end,:)' ;
ids = find(rnn_i(n+1:end)<0) ;
rnn_i(n+ids) = 0 ;

% objective value
wk = rnn_i(1:n);
gammak = rnn_i(n+1:2*n) ;
sigmak = rnn_i(end) ;
obj_val = obj_SBL(y, Phi, wk, gammak, sigmak, paras);

xhat = rnn_i(1:n) ;
if nmd ==1
    xhat = xhat * y_max ./Phi_norm' ;
end
xhat(abs(xhat)/norm(xhat)<threshold) = 0 ;
end


%% objective function
function L = obj_SBL(y,Phi,w,gamma,sigma,paras)
thre = 1e-8 ;
b0 = paras.b0 ;
c1 = 2*paras.c0 - 2 ;
d0 = paras.d0 ;
a0 = paras.a0 ;
s0 = paras.a;
[m,n] = size(Phi);
n1 = n+2-m-2*a0 ;
H = norm(y-Phi*w)^2 ;
L = sum(log(sigma+s0*gamma+thre) + c1*log(gamma+thre) + (2*d0 + w.^2)./(gamma+thre) ) ....
    + (H + 2*b0)./(sigma+thre) - n1*log(sigma+thre) ;
end


%% neurodynamic model 
function dL = fun1(t, xx, epsilons, y, Phi, paras)
thre = 1e-8;
[m,n] = size(Phi);
x1 = xx(1:n) ;
x20 = xx(1*n+1:2*n) ;
x2 = x20 ;
x2(x2<0)=0 ;
x30 = xx(end) ;
x3 = x30 ;
x3(x3<0) = 0 ;
b0 = paras.b0 ;
c1 = 2*paras.c0 - 2 ;
d0 = paras.d0 ;
a0 = paras.a0 ;
s0 = paras.a;
n1 = n+2-m-2*a0 ;

% the derivative of V
dx1 =  2*x1./(x2+thre) + 2/(x3+thre)*Phi'*(Phi*x1-y) ;
dx2 = s0./(x3+s0*x2+thre) + c1./(x2+thre) - (2*d0+x1.^2)./(x2+thre).^2 ;
H = norm(y-Phi*x1)^2  ;
dx3 = sum(1./(x3+s0*x2+thre)) - (H+2*b0)/(x3+thre)^2 - n1/(x3+thre) ;

%neurodynamic model 
dL = 1/epsilons*[-dx1; -x20+x2-dx2; -x30+x3-dx3];
end