clear ;  clc ;
addpath('tools\')
rng('default')

% load data
load('Dataset\StData2.mat') ;

% experimental setting
[m, n] = size(Phi) ; 
num_method = 3 ;
nums = 100;   snr = 40 ;
filename = ['ResultsData\St_',num2str(snr),'dB_n',num2str(n),'_2NN_CM2.mat'] ;

% parameters for both methods 
paras1.a0 = 1e-6 ;       paras1.b0 = 1e-6 ;
paras1.c0 = 1+1e-6;      paras1.d0 = 1e-6 ;
paras1.e0 = 1e-2 ;
paras1.iters = 5000;     paras1.threshold = 1e-3;
paras1.delta = 1e-5 ;   % for stopping criterion
paras1.normalized = 0 ;
paras1.NN_number = 2 ;
paras1.max_iterions = 10 ;
Tstep = 1e-7 ;
paras1.Tspan = 0:Tstep:1e-5 ;
tau = 1e-4 ;
delta3 = 1e-2 ;
if paras1.normalized==0
    s0 = max(eig(Phi'*Phi)) + tau;
else
    s0 = max(eig(normc(Phi)'*normc(Phi))) + tau ;
end
paras1.a = s0;

[ids0,~] = find(w) ;

% storage the results
time_SBL = zeros(nums, num_method);
errs = zeros(nums, num_method);
num_success = zeros(1, num_method);
Iterations = zeros(nums,2);
Nzeros_num = zeros(nums,num_method) ;
Lgs = cell(nums,1);
L = zeros(nums,num_method) ;
W_hats = cell(nums,1);

fprintf(2,'The matrix has %d basis functions:\n',n) ;

% main loop
for kk = 1:nums    
    xhat = zeros(n,num_method);
    fprintf('The %d th experiment:\n',kk) ;    
    
    % SBL based on CNO
    ii = 1 ;    % index of method
    tic ;
    [xhat(:,ii),objVal] = SBL_PNN2_ode23s_CM(y_noise, Phi, paras1) ;
    time_SBL(kk,ii) = toc ;
    L(kk,ii) = objVal(end) ;
    Lgs{kk,1} = objVal ;
    Iterations(kk,1) = length(objVal) ;
    errs(kk,ii) = norm(xhat(:,ii)-w)/norm(w) ;
    [ids,~] = find (xhat(:,ii)) ; 
    if (norm(xhat(:,ii)-w,'inf')/norm(w)<=delta3) && all(ids==ids0)
        num_success(1,ii) = num_success(1,ii)+1;
    end
    Nzeros_num(kk,ii) = length(nonzeros(xhat(:,ii)));
    
    
    InitVal.beta_init = randn(n,1);
    InitVal.gamma_init= rand(1);
    InitVal.lambda_init = rand(1);
    
    
    % PNN 
    ii = ii + 1;
    tic
    [xhat(:,ii),L(kk,ii)] = PNN_ode23s(y_noise, Phi, paras1, InitVal) ;
    time_SBL(kk,ii) = toc ;
    errs(kk,ii) = norm(xhat(:,ii)-w)/norm(w) ;
    [ids,~] = find (xhat(:,ii)) ; 
    if (norm(xhat(:,ii)-w,'inf')/norm(w)<=delta3) && all(ids==ids0)
        num_success(1,ii) = num_success(1,ii)+1;
    end
    Nzeros_num(kk,ii) = length(nonzeros(xhat(:,ii))); 
    
    
    % S-ESBL
    ii = ii + 1;   
    tic
    [xhat(:,ii),Iterations(kk,2),L(kk,ii)] = Ga_FSBL(y_noise, Phi, paras1, InitVal) ;
    time_SBL(kk,ii) = toc ;
    errs(kk,ii) = norm(xhat(:,ii)-w)/norm(w) ;
    [ids,~] = find(xhat(:,ii)) ; 
    if (norm(xhat(:,ii)-w,'inf')/norm(w)<=delta3) && all(ids==ids0)
        num_success(1,ii) = num_success(1,ii)+1;
    end
    Nzeros_num(kk,ii) = length(nonzeros(xhat(:,ii)));    
        
    % the estimated signals
    W_hats{kk,1} = xhat ;
end


% the average results
err_mean = mean(errs);
err_std = std(errs,1);
time_mean = mean(time_SBL);
Nzeros_mean = mean(Nzeros_num);
Iter_mean = mean(Iterations) ;

disp('the average error is :') 
disp(err_mean) 

sound(sin(2*pi*25*(1:4000)/100));

save(filename) ;


%%   plot figures
markers = {'o','s','d','v'};
colors = {[0 0.4470 0.7410],[0.8500 0.3250 0.0980],[0.9290 0.6940 0.1250]} ;
figure()
for i=1:num_method
    scatter(1:nums,L(:,i),markers{i},'filled','MarkerEdgeColor',colors{i})
    %,'MarkerEdgeColor','k','MarkerFaceColor',colors{i});
    hold on
end
set(gca,'FontSize',12)
legend('CNO-SBL','PNN','S-ESBL')
xlabel('Sequence Number of Experimental Runs')
ylabel('Objective Functin Value')
grid on

w_est = W_hats{25,1};
figure()
subplot(4,1,1)
plot(1:n,w,'LineWidth',2)
set(gca,'FontSize',12)
ylabel('True')
xlim([1,20])
grid on 
subplot(4,1,2)
plot(1:n,w_est(:,1),'LineWidth',2)
set(gca,'FontSize',12)
ylabel('CNO-SBL')
xlim([1,20])
grid on
subplot(4,1,3)
plot(1:n,w_est(:,2),'LineWidth',2)
set(gca,'FontSize',12)
ylabel('PNN')
xlim([1,20])
ylim([-0.5, 1])
grid on
subplot(4,1,4)
plot(1:n,w_est(:,3),'LineWidth',2)
set(gca,'FontSize',12)
ylabel('S-ESBL')
xlim([1,20])
ylim([-0.5, 1.2])
grid on
xlabel('Signal Length')


% Lgi = Lgs{100,1};
% iters = length(Lgi)-1;
% figure()
% plot(0:iters,Lgi,'LineWidth',2)
% set(gca,'FontSize',12)
% xlabel('Iterations')
% ylabel('Objective Functin Value')
% % ylabel('$\widehat{L}(u_g)$','interpreter','latex')
% grid on