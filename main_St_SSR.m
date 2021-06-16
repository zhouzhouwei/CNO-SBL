clear ;  clc ;
addpath('tools\')
rng('default')

% load data
load('Dataset\StData2.mat') ;

% experimental setting
[m, n] = size(Phi) ; 
num_method = 7 ;  nums = 100;   snr = 40 ;
filename = ['ResultsData\St_',num2str(snr),'dB_n',num2str(n),'_2NN_CM.mat'] ;

% parameters for both methods 
paras.a0 = 1e-6 ;       paras.b0 = 1e-6 ;
paras.c0 = 1+1e-6;      paras.d0 = 1e-6 ;
paras.iters = 1000;     paras.threshold = 1e-3;
paras.delta = 1e-5 ;   % for stopping criterion
paras.normalized = 0 ;
paras.NN_number = 2 ;
paras.max_iterions = 10 ;
Tstep = 1e-7 ;
paras.Tspan = 0:Tstep:1e-5 ;
tau = 1e-4 ;
delta3 = 1e-2 ;
if paras.normalized==0
    s0 = max(eig(Phi'*Phi)) + tau;
else
    s0 = max(eig(normc(Phi)'*normc(Phi))) + tau ;
end
paras.a = s0;

[ids0,~] = find(w) ;

% storage the results
time_SBL = zeros(nums, num_method);
errs = zeros(nums, num_method);
num_success = zeros(1, num_method);
Iterations = zeros(nums,3);
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
    [xhat(:,ii),objVal] = SBL_PNN2_ode23s_CM(y_noise, Phi, paras) ;
    time_SBL(kk,ii) = toc ;
    L(kk,ii) = objVal(end) ;
    Lgs{kk,1} = objVal ;
    Iterations(kk,1) = length(objVal) ;
    errs(kk,ii) = norm(xhat(:,ii)-w)/norm(w) ;
    [ids,~] = find (xhat(:,ii)) ; 
    if length(ids) == length(ids0)
        if (norm(xhat(:,ii)-w,'inf')/norm(w)<=delta3) && all(ids==ids0)
            num_success(1,ii) = num_success(1,ii)+1;
        end
    end
    Nzeros_num(kk,ii) = length(nonzeros(xhat(:,ii)));
    
    
    InitVal.beta_init = randn(n,1);
    InitVal.gamma_init = rand(1);
    InitVal.lambda_init = rand(1) ;
    
    % PNN 
    ii = ii + 1;
    tic
    [xhat(:,ii),L(kk,ii)] = PNN_ode23s(y_noise, Phi, paras, InitVal) ;
    time_SBL(kk,ii) = toc ;
    errs(kk,ii) = norm(xhat(:,ii)-w)/norm(w) ;
    [ids,~] = find (xhat(:,ii)) ; 
    if length(ids) == length(ids0)
        if (norm(xhat(:,ii)-w,'inf')/norm(w)<=delta3) && all(ids==ids0)
            num_success(1,ii) = num_success(1,ii)+1;
        end
    end
    Nzeros_num(kk,ii) = length(nonzeros(xhat(:,ii))); 
    
    
    % S-ESBL
    ii = ii + 1;   
    tic
    [xhat(:,ii),Iterations(kk,2),L(kk,ii)] = Ga_FSBL(y_noise, Phi, paras, InitVal) ;
    time_SBL(kk,ii) = toc ;
    errs(kk,ii) = norm(xhat(:,ii)-w)/norm(w) ;
    [ids,~] = find(xhat(:,ii)) ; 
    if length(ids) == length(ids0)
        if (norm(xhat(:,ii)-w,'inf')/norm(w)<=delta3) && all(ids==ids0)
            num_success(1,ii) = num_success(1,ii)+1;
        end
    end
    Nzeros_num(kk,ii) = length(nonzeros(xhat(:,ii)));    
    
    
    % EM-SBL in Tipping 
    ii = ii + 1 ;   
    tic
    [xhat(:,ii), Iterations(kk,3), L(kk,ii)] = Tipping_SBL(y_noise, Phi, paras, InitVal) ;
    time_SBL(kk,ii) = toc ;
    errs(kk,ii) = norm(xhat(:,ii)-w) / norm(w) ;
    [ids,~] = find(xhat(:,ii)) ; 
    if length(ids) == length(ids0)
        if (norm(xhat(:,ii)-w,'inf')/norm(w)<=delta3) && all(ids==ids0)
            num_success(1,ii) = num_success(1,ii)+1;
        end
    end
    Nzeros_num(kk,ii) = length(nonzeros(xhat(:,ii)));    

    % IF-SBL
    ii = ii + 1 ;
    tic 
    [xhat(:,ii), L(kk,ii)]  = IFSBL(Phi, y_noise, paras, InitVal) ;
    time_SBL(kk,ii) = toc ;
    errs(kk,ii) = norm(xhat(:,ii)-w) / norm(w) ;
    [ids,~] = find(xhat(:,ii)) ; 
    if length(ids) == length(ids0)
        if (norm(xhat(:,ii)-w,'inf')/norm(w)<=delta3) && all(ids==ids0)
            num_success(1,ii) = num_success(1,ii)+1;
        end
    end
    Nzeros_num(kk,ii) = length(nonzeros(xhat(:,ii)));   
    
    % FLSBL 
    ii = ii + 1 ;
    delta_La = 1e-10 ;
    tic
    [weights, used_ids] = FastLaplace(Phi, y_noise, InitVal.lambda_init, delta_La, InitVal.gamma_init);
    time_SBL(kk,ii) = toc;
    temp = zeros(n,1);
    temp(used_ids) = weights ;
    temp(abs(temp)./norm(temp)<paras.threshold) = 0 ;
    xhat(:,ii) = temp ;
    errs(kk,ii) = norm(xhat(:,ii)-w)/norm(w) ;
    [ids,~] = find(xhat(:,ii)) ; 
    if length(ids) == length(ids0)
        if (norm(xhat(:,ii)-w,'inf')/norm(w)<=delta3) && all(ids==ids0)
            num_success(1,ii) = num_success(1,ii)+1;
        end
    end
    Nzeros_num(kk,ii) = length(nonzeros(xhat(:,ii)));
    
    % GGAMP-SBL 
    ii = ii + 1 ;
    tic 
    xhat(:,ii) = GGAMP_SBL(y_noise, Phi, paras, InitVal) ;
    time_SBL(kk,ii) = toc ;
    errs(kk,ii) = norm(xhat(:,ii)-w) / norm(w) ;
    [ids,~] = find(xhat(:,ii)) ; 
    if length(ids) == length(ids0)
        if (norm(xhat(:,ii)-w,'inf')/norm(w)<=delta3) && all(ids==ids0)
            num_success(1,ii) = num_success(1,ii)+1;
        end
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
disp('The number of success is:')
disp(num_success)

sound(sin(2*pi*25*(1:4000)/100));

save(filename) ;


%%   plot figures
% markers = {'o','s','d','>','h'};
% % colors = {[0 0.4470 0.7410],[0.8500 0.3250 0.0980],[0.9290 0.6940 0.1250]} ;
% figure()
% for i=1:5
%     scatter(1:nums,L(:,i),markers{i},'filled')
%     %,'MarkerEdgeColor','k','MarkerFaceColor',colors{i});
%     hold on
% end
% set(gca,'FontSize',12)
% legend('CNO-SBL (N=2)','PNN-SBL','S-ESBL','EM-SBL','IF-SBL')
% xlabel('Sequence number of experimental runs')
% ylabel('Objective Functin Value')
% grid on

w_est = W_hats{10,1};
figure()
subplot(4,2,1)
plot(1:n,w,'LineWidth',2)
set(gca,'FontSize',12)
legend('True')
xlim([1,20])
grid on 

subplot(4,2,2)
plot(1:n,w_est(:,1),'LineWidth',2)
set(gca,'FontSize',12)
legend('CNO-SBL')
xlim([1,20])
grid on

subplot(4,2,3)
plot(1:n,w_est(:,2),'LineWidth',2)
set(gca,'FontSize',12)
legend('PNN-SBL')
xlim([1,20])
ylim([-1.4, 1])
grid on

subplot(4,2,4)
plot(1:n,w_est(:,3),'LineWidth',2)
set(gca,'FontSize',12)
legend('S-ESBL')
xlim([1,20])
ylim([-1.4, 1])
grid on

subplot(4,2,5)
plot(1:n,w_est(:,4),'LineWidth',2)
set(gca,'FontSize',12)
legend('EM-SBL')
xlim([1,20])
grid on
axes('Position',[0.18 0.43 0.19 0.05]) 
ids = [4:19];
plot(ids,w_est(ids,4),'LineWidth',2)
ylim([-0.01 0.01])
xlim([4,19])
grid on
annotation('textbox',[0.187 0.399 0.288 0.0181],'LineWidth',1)
annotation('textarrow',[0.322 0.303] , [0.419 0.433])


subplot(4,2,6)
plot(1:n,w_est(:,5),'LineWidth',2)
set(gca,'FontSize',12)
legend('LP-SBL')
xlim([1,20])
ylim([-1.4, 1])
grid on

subplot(4,2,7)
plot(1:n,w_est(:,6),'LineWidth',2)
set(gca,'FontSize',12)
legend('LP-SBL')
xlim([1,20])
grid on
xlabel('Signal Length')

subplot(4,2,8)
plot(1:n,w_est(:,7),'LineWidth',2)
set(gca,'FontSize',12)
legend('GGAMP-SBL')
xlim([1,20])
grid on
xlabel('Signal Length')
axes('Position',[0.636 0.217 0.190 0.0436]) 
ids = [1:19];
plot(ids,w_est(ids,7),'LineWidth',2)
ylim([-0.02 0.02])
xlim([1,19])
grid on
annotation('textbox',[0.645 0.181 0.269 0.018],'LineWidth',1)
annotation('textarrow',[0.804 0.773], [0.201 0.211])


% Lgi = Lgs{100,1};
% iters = length(Lgi)-1;
% figure()
% plot(0:iters,Lgi,'LineWidth',2)
% set(gca,'FontSize',12)
% xlabel('Iterations')
% ylabel('Objective Functin Value')
% % ylabel('$\widehat{L}(u_g)$','interpreter','latex')
% grid on