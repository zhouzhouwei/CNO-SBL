clear; clc;
addpath('tools\')
MyData = load('Dataset\SGdata15points.mat');
nums = 100 ; num_method= 2;
filename = 'ResultsData\SG_CMres_15.mat' ;
rng(1) 

Dic = MyData.theta ;
Y = MyData.y ;
W = MyData.w_true;
n_variables = size(Y,2);

% parameters
paras.a0 = 1e-6;       paras.b0 = 1e-6;
paras.c0 = 1+1e-6;     paras.d0 = 1e-6 ;
paras.iters = 5000;    paras.threshold = 1e-3;
paras.delta = 1e-5 ;   % for stopping criterion
paras.normalized = 1 ;
paras.NN_number = 10 ;
paras.max_iterions = 10;
Tstep = 1e-7 ;
paras.Tspan = 0:Tstep:1e-4 ;
tau = 1e-4 ;
if paras.normalized==0
    s0 = max(eig(Dic'*Dic)) + tau;
else
    s0 = max(eig(normc(Dic)'*normc(Dic))) + tau ;
end
paras.a = s0;
[~,n] = size(Dic);

% storage the results
time_SBL = zeros(nums, n_variables, num_method);
Nzeros_num = zeros(nums,n_variables,num_method) ;
L = zeros(nums,n_variables,num_method) ;
errs = zeros(nums,num_method);
W_hats = cell(nums,2);
Lgs = cell(nums,n_variables) ;

% main loop 
for kk=1:nums
    fprintf('The %d th experiment:\n',kk) ; 
    W_est1 = zeros(size(W)) ;
    W_est2 = zeros(size(W)) ;
    for ii = 1:n_variables
        y = Y(:,ii);
        
        % method 1 
        tic 
        [W_est1(:,ii),objVal] = SBL_PNN2_ode23s_CM(y, Dic, paras) ;
        time_SBL(kk,ii,1) = toc ;
        L(kk,ii,1) = objVal(end) ;
        Lgs{kk,ii} = objVal ;
        Nzeros_num(kk,ii,1) = length(nonzeros(W_est1(:,ii)));
        
        % method 2
        InitVal.beta_init = randn(n,1);
        InitVal.gamma_init= rand(1);
        InitVal.lambda_init = rand(1);        
        tic ;
        [W_est2(:,ii), ~, L(kk,ii,2)] = Ga_FSBL(y, Dic, paras, InitVal) ;
        time_SBL(kk,ii,2) = toc ;
        Nzeros_num(kk,ii,2) = length(nonzeros(W_est2(:,ii)));
    end
    errs(kk,1) = mean(vecnorm(W_est1 - W)./vecnorm(W));
    errs(kk,2) = mean(vecnorm(W_est2 - W)./vecnorm(W));
    W_hats{kk,1} = W_est1 ;
    W_hats{kk,2} = W_est2 ;
end

% the average results
err_mean = mean(errs);
err_std = std(errs,1);
time_mean = mean(time_SBL);
time_mean = squeeze(time_mean) ;
disp('the average error is:') 
disp(err_mean);

% end tone 
sound(sin(2*pi*25*(1:4000)/100));

%save results 
save(filename) ;

%%  plot figures 
markers = {'o','s','>','v'};
figure()
for i=1:num_method
    scatter(1:nums,L(:,i),markers{i},'linewidth',2);
    hold on
end
set(gca,'FontSize',12)
legend('CNO-SBL','S-ESBL')
xlabel('Sequence Number of Experimental Runs')
ylabel('Objective Functin Value')
grid on


Lgi = Lgs{4,1};
iters = length(Lgi)-1;
figure()
plot(0:iters,Lgi,'LineWidth',2)
set(gca,'FontSize',12)
xlabel('Iterations')
ylabel('Objective Functin Value')
% ylabel('$\widehat{L}(u_g)$','interpreter','latex')
grid on
