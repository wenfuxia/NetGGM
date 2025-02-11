function [Sigma] = GenerateCliquesCovariance(nCliques, cliqueSize, value)

p = nCliques * cliqueSize;
Sigma = zeros(p, p);

for i = 0:(nCliques - 1)
    for j = (i * cliqueSize + 1):((i + 1) * cliqueSize - 1)
        for k = (j + 1):((i + 1) * cliqueSize)
            Sigma(j, k) = (randi([0,1]) - 0.5) * 2 * value;
        end
    end
end
Sigma = Sigma + Sigma';
eigenvalues = eig(Sigma);
shift = (eigenvalues(p) - p * eigenvalues(1)) / (p - 1);
Sigma = Sigma + shift * eye(p);

end