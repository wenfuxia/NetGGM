clear;figure(1);clf;figure(2);clf;figure(3);clf;clc;
max_iter = 10000000;
Is = [5, 10, 20];
taus_G = {20, 1, 0.8, 0.6; 30, 2, 1.5, 0.7; 200, 20, 10, 5};
alpha = 0.1;
alpha_o = alpha;
%m = I;
p = 50;
d = p;
%sigma = GenerateCliquesCovariance(5, p / 5, 1);
lambdas_G = [0.5, 0.2, 0.05, 0.03, 0.02, 0.02];
nu = 10;
a = 0;
b = 0;
%func = "t-distribution";
func = "Gaussian";
gen_func = "Gaussian";
ns = [25, 50, 75, 100, 125, 150];
Sigma_mean_G = cell(3, 4); 
result_G = cell(3, 4);
Obj_G = cell(3, 4); 
ds_G = cell(3, 4); 
nmse_G = cell(3, 4);
Sigma_mean_t = cell(3, 4); 
result_t = cell(3, 4);
Obj_t = cell(3, 4); 
ds_t = cell(3, 4); 
nmse_t = cell(3, 4);
minlambdas = zeros(1, 6);
Theta_opt = cell(1, 6);
Theta_bl = cell(3, 2,6);
NMSEbl = zeros(3, 2, 6);
obj_G = cell(1, 6);
NMSEc = zeros(1, 6);
Theta = cell(3, 3, 6);
Theta_L = cell(3, 3, 6);
Theta_M = cell(3, 3, 6);
optDist = cell(3, 3, 6);
gtDist = cell(3, 3, 6);
obj = cell(3, 3, 6);
conErr = cell(3, 3, 6);
%invsigma = GenerateCliquesCovariance(5, d / 5, 1);
invsigma = GenerateRandomCovariance(d, 1, 0.05);
sigma = inv(invsigma);
lambdas = {
    [1e-2:1e-2:1e-1, 2e-1:1e-1:1e0],
    [1e-2:1e-2:1e-1, 2e-1:1e-1:1e0],
    [1e-2:1e-2:1e-1, 1e-1:1e-1:1e0],
    [1e-3:1e-3:1e-2, 1e-2:1e-2:1e-1],
    [1e-2:1e-2:1e-1, 1e-1:1e-1:1e0],
    [1e-2:1e-2:1e-1, 1e-1:1e-1:1e0],
};
times = zeros(3, 6, 6);
times_L = zeros(3, 6, 6);
times_M = zeros(3, 6, 6);
for m = Is
    I = m;
    Wxs = {genNetwork(0.9, m, p), genNetwork(0.5, m, p)};
    Adj = zeros(m);
    for i = 1:m - 1
        Adj(i, i + 1) = 1;
        Adj(i + 1, i) = 1;
    end
    degree=diag(sum(Adj));  %Degree matrix
    A = zeros(I);
    for i=1:I
        i_link=find(Adj(i,:)>0);
        for j=1:I
            if i~=j && sum(find(j==i_link))>0
                A(i,j)=1/(max(degree(i,i),degree(j,j))+1);
            end
        end
    end
    W=eye(I)-diag(sum(A))+A; %Weight matrix
    Wxs{3} = kron(W, eye(p));
    save("network_W_" + int2str(m) + ".mat", 'Wxs')
end

for mtcl = 1:100
    xxs = cell(1,6);
    xs = cell(3,6);
    for iter = [1, 4]
        iter
        xx = mvnrnd(zeros(p, 1), sigma, ns(iter));
        xxs{iter} = xx;
        
        minlambda = 0;
        minTheta = 0;
        minNMSE = 1e10;
        minObj = 0;
        minTime = 0;
        for lambda = lambdas{iter}
            lambda
            tic
            [Theta_opt{iter}, obj_G{iter}] = G_ISTA(xx' * xx / ns(iter), lambda, 5e-9, 1e8);
            times(1, 1, iter) = toc;
            toc
            NMSEc(iter) = norm(Theta_opt{iter} - invsigma, 'fro') ^ 2 / norm(invsigma, 'fro') ^ 2;
            if minNMSE > NMSEc(iter)
                minNMSE = NMSEc(iter);
                minlambda = lambda;
                minTheta = Theta_opt{iter};
                minObj = obj_G{iter};
                minTime = times(1, 1, iter);
            end
        end
        minlambdas(iter) = minlambdas(iter) + minlambda / 10;
        
        lambda = minlambda;
        %[Theta_opt{iter}, obj_G{iter}] = G_ISTA(xx' * xx / ns(iter), lambda, 5e-9, 1e6);
        Theta_opt{iter} = minTheta;
        obj_G{iter} = minObj;
        NMSEc(iter) = minNMSE;
        times(1, 1, iter) = minTime;
        
        for nAgent = 1:3
            
            nAgent
            m = Is(nAgent);
            load("network_W_" + int2str(m) + ".mat", 'Wxs');
            n = rand_sum(ns(iter), m);
            while n(1) <= 0
                n = rand_sum(ns(iter), m);
            end
            x = cell(1, m);
            pointer = 1;
            for i = 1:m
                x{i} = xx(pointer:pointer + n(i) - 1, :);
                pointer = pointer + n(i);
            end
            xs{nAgent, iter} = x;
            
            lambdabl1 = zeros(1, m);
            for i = 1:m
                lambdabl1(i) = sqrt(log(d) / n(i));
            end
            tic
            Theta_bl{nAgent, 1, iter} = distributedInverseCovEst_weighted_par(x, lambdabl1, 0, sqrt(log(d) / ns(iter)), n);
            NMSEbl(nAgent, 1, iter) = norm(Theta_bl{nAgent, 1, iter} - invsigma, 'fro') ^ 2 / norm(invsigma, 'fro') ^ 2;
            times(nAgent, 5, iter) = toc;
            toc
            tic
            Theta_bl{nAgent, 2, iter} = distributed_precision_estimation_par(m, n, x, lambdabl1);
            NMSEbl(nAgent, 2, iter) = norm(Theta_bl{nAgent, 2, iter} - invsigma, 'fro') ^ 2 / norm(invsigma, 'fro') ^ 2;
            times(nAgent, 6, iter) = toc;
            toc

            for gr = 1:1:3
                gr
                W = recoverW(Wxs{gr}, d, m);
                [Laplacian_W, MaxDegree_W] = compute_weight_matrices(W);
                tau = 1;
                tic
                [Theta{nAgent, gr, iter}, optDist{nAgent, gr, iter}, gtDist{nAgent, gr, iter}, obj{nAgent, gr, iter}, conErr{nAgent, gr, iter}, check] = Network_GGM(xs{nAgent, iter}, ns(iter), m, d, lambda, tau, alpha, max_iter, 1e-7, 1e-7, minTheta, Wxs{gr});
                %[Theta{nAgent, gr, iter}, optDist{nAgent, gr, iter}, obj{nAgent, gr, iter}, check] = Network_GGM_par(xs{nAgent, iter}, ns(iter), m, d, lambda, tau, alpha, max_iter, 1e-7, 1e-7, invsigma, Wxs{gr});
                times(nAgent, gr + 1, iter) = toc;
                toc
                while check && abs(norm(kron(ones(1,m), eye(d)) * Theta{nAgent, gr, iter} / m - invsigma, 'fro') ^ 2 / norm(invsigma, 'fro') ^ 2 - minNMSE) >= 1e-8
                    if nAgent == 1
                        if gr == 1
                            tau = 1.25 * tau;
                        elseif gr == 2
                            tau = 1.5 * tau;
                        else
                            tau = 1.75 * tau;
                        end
                    elseif nAgent == 2
                        if gr == 1
                            tau = 1.5 * tau;
                        elseif gr == 2
                            tau = 1.75 * tau;
                        else
                            tau = 2 * tau;
                        end
                    else
                        if gr == 1
                            tau = 2 * tau;
                        elseif gr == 2
                            tau = 2.25 * tau;
                        else
                            tau = 2.5 * tau;
                        end
                    end
                    
                    tic
                    [Theta{nAgent, gr, iter}, optDist{nAgent, gr, iter}, gtDist{nAgent, gr, iter}, obj{nAgent, gr, iter}, conErr{nAgent, gr, iter}, check] = Network_GGM(xs{nAgent, iter}, ns(iter), m, d, lambda, tau, alpha, max_iter, 1e-7, 1e-7, minTheta, Wxs{gr});
                    %[Theta{nAgent, gr, iter}, optDist{nAgent, gr, iter}, obj{nAgent, gr, iter}, check] = Network_GGM_par(xs{nAgent, iter}, ns(iter), m, d, lambda, tau, alpha, max_iter, 1e-7, 1e-7, invsigma, Wxs{gr});
                    times(nAgent, gr + 1, iter) = toc;
                    toc
                end
                
                tau = 1;
                tic
                [Theta_L{nAgent, gr, iter}, ~, ~, ~, ~, check] = Network_GGM(xs{nAgent, iter}, ns(iter), m, d, lambda, tau, alpha, max_iter, 1e-7, 1e-7, minTheta, kron(Laplacian_W, eye(d)));
                %[Theta{nAgent, gr, iter}, optDist{nAgent, gr, iter}, obj{nAgent, gr, iter}, check] = Network_GGM_par(xs{nAgent, iter}, ns(iter), m, d, lambda, tau, alpha, max_iter, 1e-7, 1e-7, invsigma, Wxs{gr});
                times_L(nAgent, gr, iter) = toc;
                toc
                while check && abs(norm(kron(ones(1,m), eye(d)) * Theta_L{nAgent, gr, iter} / m - invsigma, 'fro') ^ 2 / norm(invsigma, 'fro') ^ 2 - minNMSE) >= 1e-8
                    if nAgent == 1
                        if gr == 1
                            tau = 1.25 * tau;
                        elseif gr == 2
                            tau = 1.5 * tau;
                        else
                            tau = 1.75 * tau;
                        end
                    elseif nAgent == 2
                        if gr == 1
                            tau = 1.5 * tau;
                        elseif gr == 2
                            tau = 1.75 * tau;
                        else
                            tau = 2 * tau;
                        end
                    else
                        if gr == 1
                            tau = 2 * tau;
                        elseif gr == 2
                            tau = 2.25 * tau;
                        else
                            tau = 2.5 * tau;
                        end
                    end
                    
                    tic
                    [Theta_L{nAgent, gr, iter}, ~, ~, ~, ~, check] = Network_GGM(xs{nAgent, iter}, ns(iter), m, d, lambda, tau, alpha, max_iter, 1e-7, 1e-7, minTheta, kron(Laplacian_W, eye(d)));
                    %[Theta{nAgent, gr, iter}, optDist{nAgent, gr, iter}, obj{nAgent, gr, iter}, check] = Network_GGM_par(xs{nAgent, iter}, ns(iter), m, d, lambda, tau, alpha, max_iter, 1e-7, 1e-7, invsigma, Wxs{gr});
                    times_L(nAgent, gr, iter) = toc;
                    toc
                end
                
                tau = 1;
                tic
                [Theta_M{nAgent, gr, iter}, ~, ~, ~, ~, check] = Network_GGM(xs{nAgent, iter}, ns(iter), m, d, lambda, tau, alpha, max_iter, 1e-7, 1e-7, minTheta, kron(MaxDegree_W, eye(d)));
                %[Theta{nAgent, gr, iter}, optDist{nAgent, gr, iter}, obj{nAgent, gr, iter}, check] = Network_GGM_par(xs{nAgent, iter}, ns(iter), m, d, lambda, tau, alpha, max_iter, 1e-7, 1e-7, invsigma, Wxs{gr});
                times_M(nAgent, gr, iter) = toc;
                toc
                while check && abs(norm(kron(ones(1,m), eye(d)) * Theta_M{nAgent, gr, iter} / m - invsigma, 'fro') ^ 2 / norm(invsigma, 'fro') ^ 2 - minNMSE) >= 1e-8
                    if nAgent == 1
                        if gr == 1
                            tau = 1.25 * tau;
                        elseif gr == 2
                            tau = 1.5 * tau;
                        else
                            tau = 1.75 * tau;
                        end
                    elseif nAgent == 2
                        if gr == 1
                            tau = 1.5 * tau;
                        elseif gr == 2
                            tau = 1.75 * tau;
                        else
                            tau = 2 * tau;
                        end
                    else
                        if gr == 1
                            tau = 2 * tau;
                        elseif gr == 2
                            tau = 2.25 * tau;
                        else
                            tau = 2.5 * tau;
                        end
                    end
                    
                    tic
                    [Theta_M{nAgent, gr, iter}, ~, ~, ~, ~, check] = Network_GGM(xs{nAgent, iter}, ns(iter), m, d, lambda, tau, alpha, max_iter, 1e-7, 1e-7, minTheta, kron(MaxDegree_W, eye(d)));
                    %[Theta{nAgent, gr, iter}, optDist{nAgent, gr, iter}, obj{nAgent, gr, iter}, check] = Network_GGM_par(xs{nAgent, iter}, ns(iter), m, d, lambda, tau, alpha, max_iter, 1e-7, 1e-7, invsigma, Wxs{gr});
                    times_M(nAgent, gr, iter) = toc;
                    toc
                end
            end
        end
        
    end
    name = "NetGGM_mtcl=" + int2str(mtcl)
    save(name + ".mat", 'Theta_opt', 'Theta_bl', 'Theta', 'Theta_L', 'Theta_M', 'times', 'times_L', 'times_M')
end

function [random_integers] = rand_sum(total_sum, num_elements)

    random_values = rand(1, num_elements - 1);

    sorted_values = sort(random_values);

    partition_points = [0, sorted_values, 1];

    diff_values = diff(partition_points);

    random_integers = round(diff_values * (total_sum - num_elements)) + 1;

    adjustment = total_sum - sum(random_integers);
    random_integers(1) = random_integers(1) + adjustment;

end

function [Laplacian_W, MaxDegree_W] = compute_weight_matrices(W)

    m = size(W, 1);
    Adj = (W ~= 0);
    Adj = Adj - diag(diag(Adj));
    
    degrees = sum(Adj, 2) + 1;
    max_degree = max(degrees);
    
    Laplacian_W = zeros(m);
    MaxDegree_W = zeros(m);
    
    Laplacian_W(Adj == 1) = 1 / max_degree;
    MaxDegree_W(Adj == 1) = 1 / m;
    
    Laplacian_W = Laplacian_W + eye(m) - diag(sum(Laplacian_W, 2));
    MaxDegree_W = MaxDegree_W + eye(m) - diag(sum(MaxDegree_W, 2));
end


function W = recoverW(Wx, d, m)
    % Initialize the matrix W
    W = zeros(m, m);
    
    % Iterate through each block to recover W from Wx
    for i = 1:m
        for j = 1:m
            % Block indices
            row_idx = (i - 1) * d + 1;
            col_idx = (j - 1) * d + 1;
            
            % Recover W(i, j) from the top-left element of the block
            W(i, j) = Wx(row_idx, col_idx);
        end
    end
    
    return;
end
