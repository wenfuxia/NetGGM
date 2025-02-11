function [Wx] = genNetwork(pp, I, p)
%%%%%%%%%%%%% network %%%%%%%%%%%

while true
    pErdosRenyi = pp;
    p_data_gen = pErdosRenyi;
    
    Adj = rand(I) < p_data_gen;
    Adj = triu(Adj);
    Adj = Adj + Adj';
    
    %Adj = randraw('bernoulli',pErdosRenyi,I,I);
    
    I_NN = eye(I);
    NotI = ~I_NN;
    
    Adj = Adj.*NotI; % eliminate zeros on diag
    
    Adj = or(Adj,Adj'); % symmetrize, undirected
    

    testAdj = (I_NN+Adj)^I; % check if G is connected
    
    if ~any(any(~testAdj))
        fprintf('The ER graph is connected\n');
        break;
    else
        % fprintf('The graph is disconnected\n');
    end
end

%%%% MH weight %%%%
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
Wx = kron(W, eye(p));
end