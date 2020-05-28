function [xhat, obj_val, xhat_ind] = SBL_PNN2_ode23s_CM(y, Phi, paras)
NN_number = paras.NN_number  ;
max_iterations = paras.max_iterions ;
threshold = paras.threshold;
tspan =  paras.Tspan ;
epsilons = 1e-6;  % time constant 
tau = 1e-10 ;
delta = 0.2;
[~,n] = size(Phi) ;
nmd = paras.normalized ;
if nmd ==1 
    y_max = max(abs(y)) ;
    y = y/ y_max ;
    Phi_norm = vecnorm(Phi) ;
    Phi = Phi ./ vecnorm(Phi) ;
end

%initilization for all NNs
w_init = randn(n,NN_number)  ;
gamma_init = ones(n,1)*rand(1,NN_number) ;
sigma_init = rand(1,NN_number) ;
u0 = [w_init; gamma_init; sigma_init];
v0 = rand(size(u0));
u_Gbest = zeros(2*n+2,max_iterations+1) ;

L0  = zeros(1,NN_number) ;
for i=1:NN_number
    L0(1,i) = obj_SBL(y, Phi, w_init(:,i), gamma_init(:,i), sigma_init(:,i), paras) ;
end
[~,idx] = min(L0) ;
u_Pbest = [u0 ; L0 ] ;  % individual best position 
u_Gbest(:,1) = [u0(:,idx); L0(idx)] ;
Lk  = zeros(max_iterations+1, NN_number) ;
Lk(1,:) = L0 ;
Pg_flag=zeros(max_iterations+1,1);
results_all = cell(max_iterations,1);

% main loop
for count = 1:max_iterations
    rnn_steady = zeros(2*n+1,NN_number);
    
    % update the state of each NN
    for i=1:NN_number
        u0i  = u0(:,i); % initial state
        
        % neurodynamic model 
        [t,ut] = ode23s(@(t,xx) fun1(t, xx, epsilons, y, Phi, paras), tspan, u0i) ;       
        
        % output equation via projection 
        rnn_i = ut(end,:)' ;
        ids = find(rnn_i(n+1:end)<0) ;
        rnn_i(n+ids) = 0 ;
        
        % objective value 
        wk = rnn_i(1:n);
        gammak = rnn_i(n+1:2*n) ;
        sigmak = rnn_i(end) ;
        Lk(count+1,i) = obj_SBL(y, Phi, wk, gammak, sigmak, paras);
        
        % update individual best position Pbest 
        if Lk(count+1,i) < u_Pbest(end,i)
            u_Pbest(1:end-1,i) = rnn_i ;
            u_Pbest(end,i) = Lk(count+1,i) ;
        end

        % store all rnn results
        rnn_steady(:,i) = rnn_i ;
    end
    results_all{count,1}=rnn_steady ;
    
    % update global best position Gbest 
    [~,idx] = min(Lk(count+1,:)) ;
    if Lk(count+1,idx) < u_Gbest(end,count)
        u_Gbest(1:end-1,count+1) = rnn_steady(:,idx) ;
        u_Gbest(end,count+1) = Lk(count+1,idx) ;
    else
        u_Gbest(:,count+1) = u_Gbest(:,count);
    end
    
    % cauchy mutuation
    qn = mean(vecnorm(u_Pbest(1:end-1,:)-u_Gbest(1:end-1,count+1))) ;
    if qn < delta
        rnn_steady = rnn_steady + rnn_steady.*trnd(1,size(rnn_steady)) ;       
    end
    
    % update the next initial state based on PSO
    for j = 1:NN_number
        [Xtemp,Vtemp] = PSO(v0(:,j), rnn_steady(:,j), u_Pbest(1:end-1,j), u_Gbest(1:end-1,count+1));
        u0(:,j) = Xtemp ;  
        v0(:,j) = Vtemp ;
    end
    
    % termination
    if norm(u_Gbest(1:end-1,count)-u_Gbest(1:end-1,count+1))<tau
        Pg_flag(count,1) = 1;
        if count>=5
            ids = find(~Pg_flag(1:count+1));
            cond_val = max(ids(2:end)-ids(1:end-1)-1);
            if cond_val >=5
                break;
            end
        end
    end
end

% figure()
% plot(1:count, Lk(2:count+1,:),'linewidth',2);
% figure()
% plot(1:count, u_Gbest(end,2:count+1),'linewidth',2);

obj_val = u_Gbest(end,1:count+1) ;
xhat = u_Gbest(1:n, count+1) ;
xhat_ind = u_Pbest(1:n,:);

if nmd ==1
    xhat = xhat * y_max ./Phi_norm' ;
    xhat_ind = xhat_ind * y_max ./ Phi_norm' ;
end
xhat(abs(xhat)/norm(xhat)<threshold) = 0 ;
xhat_ind(abs(xhat_ind)./vecnorm(xhat)<threshold) = 0;
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


%%  PSO algorithm 
function [Xnext, Vnext] = PSO(V, Xbar, Xp, Xg)
varrho = 0.5 ;
eta1 = 0.6 ;
eta2 = 0.8 ;
e1 = rand(1) ;
e2 = rand(1) ;

Vnext =  varrho*V + eta1*e1*(Xp-Xbar) + eta2*e2*(Xg-Xbar) ;
Xnext = Xbar + Vnext ;

% projection on feasible region 
n = (length(Xbar)-1)/2 ;
ids = find(Xnext(n+1:end)<0) ;
Xnext(n+ids) = 0 ;
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