function consistentRunidx= RunSelection_crossISIidx(W)
%% Select the most consistent run based on Cross ISI.
% Input:
% W - demixing matrices of different runs, with dimension as the number of components x the number of
% components x the number of runs.
% Outputs:
% consistentRun_idx - index of the run with the lowest average pairwise
% cross ISI values
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code by Chunying Jia. 
% Reference:
% Q. Long, C. Jia, Z. Boukouvalas, B. Gabrielson, D. Emge, and T. Adali, "CONSISTENT RUN SELECTION FOR INDEPENDENT COMPONENT ANALYSIS:
% APPLICATION TO FMRI ANALYSIS", 2018 IEEE International Conference on Acoustics, Speech and Signal Processing.
% Please contact chunyin1@umbc.edu or qunfang1@umbc.edu.
N_run = size(W,3);
crossISIW = zeros(N_run,1);

for i = 1:N_run    
    crossISI_ij = 0;
    for j = 1:N_run        
      if j==i
      continue;
      end
      crossISI_ij = crossISI_ij + calculate_ISI(W(:,:,j),inv(W(:,:,i)));
    end    
   crossISIW(i,1) = crossISI_ij / (N_run -1);     
end
[~,consistentRunidx] = min(crossISIW); 
end

function ISI = calculate_ISI(W,A)

p = W*A;
p = abs(p);

N = size(p,1);

b1 = 0;
for i = 1:N
    a1 = 0;
    max_pij = max(p(i,:));
    for j = 1:N
        
        pij = p(i,j);
        a1 = pij/max_pij + a1;
    end
    b1 = a1 - 1 + b1;
end 

b2 = 0;
for j = 1:N
    a2 = 0;
    max_pij = max(p(:,j));
    for i = 1:N
        
        pij = p(i,j);
        a2 = pij/max_pij + a2;
    end
    b2 = a2 - 1 + b2;
end 

ISI = (b1+b2)/((N-1)*N*2);

end