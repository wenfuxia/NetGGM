function [Sigma] = GenerateRandomCovariance(p, value, prob)

Sigma = zeros(p, p);

for i = 1:(p - 1)
    for j = (i + 1):p
        x = rand();
        if x <= prob
            Sigma(i, j) = (randi([0,1]) - 0.5) * 2 * value;
        end
    end
end
Sigma = Sigma + Sigma';
eigenvalues = eig(Sigma);
shift = (eigenvalues(p) - p * eigenvalues(1)) / (p - 1);
Sigma = Sigma + shift * eye(p);

end