function Theta_final = distributedInverseCovEst_weighted_par(X, lambda, rho, tau, n_samples)
    % Distributed Inverse Covariance Estimation using Debiased Graphical Lasso
    % with weighted averaging based on varying sample sizes.
    % Input: 
    %   X: Cell array where each cell contains data matrix Xm (n x p) from machine m
    %   lambda: Array of penalty parameters for graphical lasso for each machine
    %   rho: Threshold for machine-level sparse estimator
    %   tau: Threshold for the final sparse estimator
    %   n_samples: Vector containing the number of samples on each machine
    % Output:
    %   Theta_final: Final thresholded sparse inverse covariance estimator
    
    % Initialization
    M = length(X);           % Number of machines
    p = size(X{1}, 2);       % Dimensionality (number of variables)
    Theta_hat_temp = cell(M, 1);  % Store debiased estimators from each machine
    Sigma_hat_temp = cell(M, 1);  % Covariance estimators from each machine
    
    % Start parallel pool if not already started
    if isempty(gcp('nocreate'))
        parpool; % Start a new pool of parallel workers if no pool exists
    end
    
    % Step 1: On each machine - Parallel computation using parfor
    parfor m = 1:M
        % Get data from machine m
        Xm = X{m};
        n = n_samples(m); % Number of samples on machine m
        
        % Estimate covariance matrix
        Sigma_local = (1/n) * (Xm' * Xm);
        
        % Graphical lasso estimator for each machine with varying lambda
        Theta_local = graphicalLasso(Sigma_local, lambda(m));
        
        % Debiased estimator using the formula: Theta_hat_d = Theta_hat + Theta_hat * (inv(Theta_hat) - Sigma_hat) * Theta_hat;
        Theta_local_d = Theta_local + Theta_local * (inv(Theta_local) - Sigma_local) * Theta_local;
        
        % Estimate variance for thresholding
        sigma_hat_local = zeros(p, p);  % Preallocate memory for sigma_hat
        for i = 1:p
            for j = 1:p
                % Calculate the variance for each element (i, j)
                sigma_hat_local(i, j) = sqrt(Theta_local(i, i) * Theta_local(j, j) + Theta_local(i, j)^2);
            end
        end

        % Apply thresholding on the debiased estimator
        Theta_local_d = (abs(Theta_local_d) > rho * sigma_hat_local) .* Theta_local_d;
        
        % Store results in temporary variables
        Theta_hat_temp{m} = Theta_local_d;
    end

    % Convert the temporary variables back to standard variables
    Theta_hat = Theta_hat_temp;

    % Step 2: In Central Hub - Weighted Aggregation and Final Thresholding
    Theta_avg = zeros(p, p); % Averaged estimator
    total_samples = sum(n_samples); % Total number of samples across all machines
    
    % Weighted average based on the sample size of each machine
    for m = 1:M
        Theta_avg = Theta_avg + (n_samples(m) / total_samples) * Theta_hat{m};
    end
    
    % Estimate variance for final thresholding
    sigma_hat_M = zeros(p, p);  % Preallocate memory for final variance
    for i = 1:p
        for j = 1:p
            sigma_hat_M(i, j) = sqrt(Theta_avg(i, i) * Theta_avg(j, j) + Theta_avg(i, j)^2);
        end
    end
    
    % Final thresholding
    Theta_final = zeros(p, p);
    Theta_final = (abs(Theta_avg) > tau * sigma_hat_M) .* Theta_avg;
end

function Theta = graphicalLasso(Sigma, lambda)
    % Placeholder function for graphical lasso implementation.
    % In practice, this could use a package or a custom implementation.
    % For example, MATLAB's `lasso` or third-party implementations can be used.
    
    % Inverse of covariance estimate as a placeholder (for illustration)
    [Theta, ~] = G_ISTA_off(Sigma, lambda, 5e-9, 1e6);
end
