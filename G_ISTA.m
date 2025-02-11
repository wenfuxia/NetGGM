function [Theta_opt, obj] = G_ISTA(S, rho, epsilon, maxIter)
    % C-ISTA with Barzilai-Borwein step for Sparse Inverse Covariance Estimation
    % Inputs:
    % S - Sample covariance matrix
    % rho - Regularization parameter
    % epsilon - Convergence tolerance
    % maxIter - Maximum number of iterations
    
    % Outputs:
    % Theta_opt - The estimated sparse inverse covariance matrix
    
    % Initialize parameters
    [p, ~] = size(S);
    Theta = inv(diag(diag(S)) + rho * eye(p));  % Initial estimate of the inverse covariance matrix
    Theta_old = Theta;              % To store previous iterate for convergence check
    step_gap = 1;
    step_size = 1;                  % Initial step size for gradient descent
    c = 0.5;                        % Backtracking constant
    %duality_gap = 2 * epsilon;      % Initialize duality gap
    
    % Iteration counter
    iter = 0;
    %obj = [- log(det(Theta)) + trace(S * Theta) + rho * sum(sum(abs(Theta))) - rho * sum(abs(diag(Theta)))];
    obj = [-log(det(Theta)) + trace(S * Theta) + rho * sum(abs(Theta(:)))];
    
    % Soft-thresholding function
    %soft_threshold = @(X, lambda) arrayfun(@(i,j) ...
    %    (i~=j) * sign(X(i,j)) * max(abs(X(i,j)) - lambda, 0) + (i==j) * X(i,j), ...
    %    repmat((1:p)',1,p), repmat(1:p,p,1));
    
    soft_threshold = @(X, lambda) sign(X) .* max(abs(X) - lambda, 0);
    
    while step_gap > epsilon && iter < maxIter
        I = eye(p);
        % Gradient step: Compute the gradient of the smooth part
        invTheta = I / Theta;
        grad = S - invTheta;
        
        % Perform backtracking line search with initial step size using Barzilai-Borwein
        while true
            % Compute next estimate using soft-thresholding
            Theta_new = soft_threshold(Theta - step_size * grad, step_size * rho);
            
            % Check positive definiteness of Theta_new
            if all(eig(Theta_new) > 0) && ...
               -log(det(Theta_new)) + trace(S * Theta_new) <= ...
               -log(det(Theta)) + trace(S * Theta) + trace(grad' * (Theta_new - Theta)) + (1 / (2 * step_size)) * norm(Theta_new - Theta, 'fro')^2
                break;
            end
            % If not, reduce step size and repeat
            step_size = c * step_size;
        end
        
        % Update step size using Barzilai-Borwein step formula (16)
        diff_Theta = Theta_new - Theta;
        diff_grad_inv = invTheta - I / Theta_new;
        numerator = trace(diff_Theta' * diff_Theta);
        denominator = trace(diff_Theta' * diff_grad_inv);
        if denominator > 0
            step_size = numerator / denominator;
        else
            step_size = 1;  % Fallback if the denominator is non-positive
        end
        
        % Update Theta
        Theta = Theta_new;
        
        % Check for convergence based on the duality gap
        %U_t_plus_1 = min(max(inv(Theta_new) - S, -rho), rho);  % Compute U_{t+1}
        %duality_gap = -log(det(S + U_t_plus_1)) - p - log(det(Theta_new)) + trace(S * Theta_new) + rho * sum(sum(abs(Theta_new))) - rho * sum(diag(Theta_new));
        step_gap = norm(Theta - Theta_old, 'fro');
        Theta_old = Theta;
        
        % Increment iteration count
        iter = iter + 1;
        %obj(iter + 1) = - log(det(Theta_new)) + trace(S * Theta_new) + rho * sum(sum(abs(Theta_new))) - rho * sum(abs(diag(Theta_new)));
        obj(iter + 1) = -log(det(Theta)) + trace(S * Theta) + rho * sum(abs(Theta(:)));
    end
    
    % Return the final estimate of the inverse covariance matrix
    Theta_opt = Theta;
end
