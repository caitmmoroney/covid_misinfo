% ICA Word Embeddings

% Load data
load('wcm_pmi_optPreprocess.mat'); % loads wcm (word-context matrix)
% NOTE: this matrix has already been standardized using scikit-learn
% StandardScaler() (zero mean, unit variance)

% NOTE: PCA NEEDS X OF SHAPE SAMPLES BY FEATURES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Run PCA on word-context matrix (shape target words by context words)
n_comp = 250;
start_pca = now();
start_pca = datetime(start_pca, 'ConvertFrom', 'datenum');
[wcm_250, wcm_250_map] = pca(wcm, 250);
end_pca = now();
end_pca = datetime(end_pca, 'ConvertFrom', 'datenum');
pca_time = end_pca - start_pca;
pca_time = seconds(pca_time);

% Get K, the pre-whitening matrix that projects data onto first n_comps
% principal components.
K = wcm_250_map.M; % K is shape M context words by 250=N components

% NOTE: ICA NEEDS X OF SHAPE COMPONENTS BY SAMPLES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Take transpose of PCA output to use as input for ICA
wcm_250_T = transpose(wcm_250); % shape: 250=N by M target words

% Run ICA (num_trials) times
num_trials = 30;
ica_time = zeros(num_trials, 1);
all_W = zeros(n_comp, n_comp);

for i = 1:num_trials
    start_time = now();
    start_time = datetime(start_time, 'ConvertFrom', 'datenum');
    W = ICA_EBM(wcm_250_T);
    end_time = now();
    end_time = datetime(end_time, 'ConvertFrom', 'datenum');
    exec_time = end_time - start_time;
    exec_time = seconds(exec_time);
    
    ica_time(i) = exec_time;
    all_W(:,:,i) = W;
%     if i == 1
%         all_W = W;
%     else
%         all_W(:,:,i) = W;
%     end
end

% Pick most representative ICA run using cross ISI
start_crossISI = now();
start_crossISI = datetime(start_crossISI, 'ConvertFrom', 'datenum');
opt_W_idx = RunSelection_crossISIidx(all_W);
end_crossISI = now();
end_crossISI = datetime(end_crossISI, 'ConvertFrom', 'datenum');
crossISI_time = end_crossISI - start_crossISI;
crossISI_time = seconds(crossISI_time);
opt_W = all_W(:,:,opt_W_idx);
save('opt_W.mat', 'opt_W') % save optimal W matrix (just in case)

% Store S, A matrices (given best W) in mat files

% S = W*K^T*X^T, where W is 250=N x 250=N, K is M context words x 250=N,
% X is M target words x M context words -> S is 250=N x M target words
% S = opt_W*transpose(K)*transpose(wcm); where mappedX = X*K
S = opt_W*wcm_250_T;
save('S.mat', 'S')

% A = pseudoinv(K^T)*(W^-1), where K is M context words x 250=N,
% W is 250=N x 250=N -> A is M context words x 250=N
K_T = transpose(K);
%A = pinv(K_T)*inv(opt_W);
A = pinv(K_T)/opt_W;
save('A.mat', 'A')


% Store execution time vals in csv file(s)
ex_times = zeros(1,3);
ex_times(1) = pca_time;
ex_times(2) = mean(ica_time);
ex_times(3) = crossISI_time;
time_table = array2table(ex_times, ...
    'VariableNames', ...
    {'PCA', 'AvgICA', 'crossISI'});
writetable(time_table, 'exec_times.csv')

ica_time_table = array2table(ica_time, ...
    'VariableNames', {'ICA'});
writetable(ica_time_table, 'ica_exec_times.csv')

