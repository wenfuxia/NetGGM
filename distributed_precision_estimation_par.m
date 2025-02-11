function Theta_hat = distributed_precision_estimation_par(K, nk, X, lambda)
    % This function implements the distributed precision matrix estimation with parallel computing
    % using parfor, including the correct calculation of sigma_hat and debiased estimates.
    %
    % Inputs:
    %   K         - Number of machines
    %   nk        - Array of sample sizes for each machine
    %   X         - Cell array of data matrices for each machine (each X{k} is an nk x p matrix)
    %   lambda    - Array of regularization parameters for each machine
    %
    % Output:
    %   Theta_hat - Estimated precision matrix

    % Number of variables
    p = size(X{1}, 2);
    
    % Initialize estimator
    Theta_hat = zeros(p, p);
    
    % Step 1: Parallel computation of local estimators on each machine using parfor
    Theta_k = cell(K, 1);      % To store local precision matrices for each machine
    sigma_hat = cell(K, 1);    % To store sigma_hat for each machine

    % Start parallel computation using parfor
    parfor k = 1:K
        % Estimate local precision matrix using graphical lasso on sub-sample k
        S_k = (X{k}' * X{k}) / nk(k);  % Empirical covariance matrix

        % Compute graphical lasso and then debiased estimator using formula (2)
        Theta_temp = graphical_lasso(S_k, lambda(k));
        Theta_k{k} = 2 * Theta_temp - Theta_temp * S_k * Theta_temp;  % Debiased estimate
        Theta_k{k} = Theta_temp;

        % Calculate sigma_hat{k} for each element (a,b)
        sigma_hat_local = zeros(p, p);
        for a = 1:p
            for b = 1:p
                % Correct calculation: sum of product of diagonal elements and square of off-diagonal element
                sigma_hat_local(a, b) = sqrt(Theta_k{k}(a, a) * Theta_k{k}(b, b) + Theta_k{k}(a, b)^2);
            end
        end
        sigma_hat{k} = sigma_hat_local;  % Assign local sigma_hat to global cell array
    end

    % Step 2: Aggregate local estimators using weights based on sample size and variance
    weight_sum = zeros(p, p);
    for k = 1:K
        for a = 1:p
            for b = 1:p
                weight = nk(k) / sigma_hat{k}(a, b)^2;
                Theta_hat(a, b) = Theta_hat(a, b) + weight * Theta_k{k}(a, b);
                weight_sum(a, b) = weight_sum(a, b) + weight;
            end
        end
    end
    
    % Final step: Normalize by the sum of weights for both off-diagonal and diagonal elements
    for a = 1:p
        for b = 1:p
            Theta_hat(a, b) = Theta_hat(a, b) / weight_sum(a, b);
        end
    end
end

function Theta = graphical_lasso(S, lambda)
    % A simple implementation of the graphical lasso algorithm using coordinate descent
    % For more advanced versions, consider using packages such as QUIC or similar
    %
    % Inputs:
    %   S       - Empirical covariance matrix
    %   lambda  - Regularization parameter
    %
    % Output:
    %   Theta   - Estimated precision matrix
    
    [Theta, ~] = G_ISTA_off(S, lambda, 5e-9, 1e6);  % Calling the existing G_ISTA function
end
