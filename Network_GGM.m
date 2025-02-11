function [Theta, optDist, gtDist, obj, conErr, check] = Network_GGM(Xs, N, m, d, lambda, tau, alpha, maxIter, tol_dis, tol_con, groundTruth, Wx)
    check = false;
    Theta = zeros(m * d, d);
    Y = zeros(m * d, d);
    Ss = zeros(m * d, d);
    ns = zeros(1, m);
    S = zeros(d);
    for i = 1:m
        idx = (i - 1) * d + 1:i * d;
        ns(i) = size(Xs{i}, 1);
        X_i = Xs{i};
        Ss(idx, :) = (X_i' * X_i) / ns(i);
        S = S + Ss(idx, :) * ns(i) / N;
        Theta(idx, :) = inv(diag(diag(Ss(idx, :))) + lambda * eye(d));
    end
    %allTheta = {Theta};
    meanTheta = kron(ones(1, m), eye(d)) * Theta / m;
    normGroundTruth = norm(groundTruth, 'fro') ^ 2;
    gtDist(1) = norm(meanTheta - groundTruth, 'fro') ^ 2 / normGroundTruth;
    conErr(1) = norm(Theta - kron(ones(m, 1), meanTheta), 'fro') ^ 2;
    obj(1) = calObj(meanTheta, S, lambda);
    Y = calDf(Theta, Ss, ns, N, m, d);
    oldDf = Y;
    
    k = 2;
    while k <= maxIter
        midTheta1 = Theta - m / tau * Y;
        midTheta2 = SoftThresholding(midTheta1, lambda / tau);
        newTheta = Wx * (Theta + alpha * (midTheta2 - Theta));
        newDf = calDf(newTheta, Ss, ns, N, m, d);
        Y = Wx * (Y + newDf - oldDf);
        
        meanTheta = kron(ones(1, m), eye(d)) * newTheta / m;
        if norm(newTheta - Theta, 'fro') <= tol_dis %&& norm(newTheta - kron(ones(m, 1), eye(d)) * meanTheta, 'fro') ^ 2 <= tol_con
            Theta = newTheta;
            k = k + 1;
            %allTheta{k} = Theta;
            break;
        end
        
        obj(k) = calObj(meanTheta, S, lambda);
        if ~isreal(obj(k)) || obj(k) > obj(k - 1)
            check = true;
            optDist = zeros(1, k);
            return;
        end

        oldDf = newDf;
        Theta = newTheta;
        
        k = k + 1;
        %allTheta{k} = Theta;
        %meanTheta = kron(ones(1, m), eye(d)) * Theta / m;
        %gtDist(k) = norm(meanTheta - groundTruth, 'fro') ^ 2 / normGroundTruth;
        %conErr(k) = norm(Theta - kron(ones(m, 1), meanTheta), 'fro') ^ 2;
    end
    optDist = zeros(1, k);
%     for l = 1:k
%         optDist(l) = norm(allTheta{l} - allTheta{k}, 'fro') ^ 2;
%     end
end

function [Sigma] = SoftThresholding(A, tau)
    Sigma = sign(A) .* max(0, abs(A) - tau);
    %Sigma = Sigma - diag(diag(Sigma)) + diag(diag(A));
end

function [Df] = calDf(Theta, Ss, ns, N, m, d)
    Df = zeros(m * d, d);
    for i = 1:m
        idx = (i - 1) * d + 1:i * d;
        Df(idx, :) = (ns(i) / N) * (Ss(idx, :) - inv(Theta(idx, :)));
    end
end

function [objective] = calObj(meanTheta, S, lambda)
    objective = - log(det(meanTheta)) + trace(S * meanTheta) + lambda * norm(meanTheta(:), 1);
end