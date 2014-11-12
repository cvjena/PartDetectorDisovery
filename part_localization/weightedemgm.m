function [label, model, llh] = weightedemgm(X, init, weights, iterations, max_iter)
    % Perform EM algorithm for fitting the Gaussian mixture model.
    %   X: d x n data matrix
    %   init: k (1 x 1) or label (1 x n, 1<=label(i)<=k) or center (d x k)
    %   weights: Must sum to N
    % Written by Michael Chen (sth4nth@gmail.com).
    label_best=0;
    model_best=0;
    llh_best=-inf;
    for i=1:iterations
        [label, model, llh] = weightedemgm_inner(X, init, weights, max_iter);
        if (llh>llh_best)
            label_best=label;
            model_best=model;
            llh_best=llh;
        end
    end
    label=label_best;
    model=model_best;
    llh=llh_best;
end

function [label, model, llh] = weightedemgm_inner(X, init, weights, iterations)
    % Perform EM algorithm for fitting the Gaussian mixture model.
    %   X: d x n data matrix
    %   init: k (1 x 1) or label (1 x n, 1<=label(i)<=k) or center (d x k)
    %   weights: Must sum to N
    % Written by Michael Chen (sth4nth@gmail.com).
    %% initialization
%     fprintf('EM for Gaussian mixture: running ... \n');
    R = initialization(X,init,weights);
    [~,label(1,:)] = max(R,[],2);
%     R = R(:,unique(label));

    tol = 1e-7;
    maxiter = iterations;
    llh = -inf(1,maxiter);
    converged = false;
    t = 1;
    while  ~converged && t < maxiter
        t = t+1;
        model = maximization(X,R);
        [R, llh(t)] = expectation(X,model,weights);

        [~,label(:)] = max(R,[],2);
        u = unique(label);   % non-empty components
        if size(R,2) ~= size(u,2)
%             R = R(:,u);   % remove empty components
%             for i=1:size(nk,2)
%                 if nk(i)==0
%                     % set mu to a random value
%                     ni=randsample(1,n)
%                     R(ni,:)=0;
%                     R(ni,i)=1;
%                 end
%             end
        else
            converged = llh(t)-llh(t-1) < tol*abs(llh(t));
        end

    end
    llh = llh(t);
    if converged
%         fprintf('Converged in %d steps.\n',t-1);
    else
%         fprintf('Not converged in %d steps.\n',maxiter);
    end
end

function R = initialization(X, init, weights)
    [d,n] = size(X);
    if isstruct(init)  % initialize with a model
        R  = expectation(X,init);
    elseif length(init) == 1  % random initialization
        k = init;
        idx = randsample(n,k);
        while length(idx)~=length(unique(idx))
            idx = randsample(n,k,true,weights);
        end
        m = X(:,idx);
        [~,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1);
        [u,~,label] = unique(label);
        while k ~= length(u)
            idx = randsample(n,k);
            m = X(:,idx);
            [~,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1);
            [u,~,label] = unique(label);
        end
        R = full(sparse(1:n,label,1,n,k,n));
    elseif size(init,1) == 1 && size(init,2) == n  % initialize with labels
        label = init;
        k = max(label);
        R = full(sparse(1:n,label,1,n,k,n));
    elseif size(init,1) == d  %initialize with only centers
        k = size(init,2);
        m = init;
        [~,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1);
        R = full(sparse(1:n,label,1,n,k,n));
    else
        error('ERROR: init is not valid.');
    end
    R = bsxfun(@times, R, weights);
    if (size(R,2)<2 || sum(isnan(R(:))>0))
        fprintf('NaN');
    end
end

function [R, llh] = expectation(X, model, weights)
    mu = model.mu;
    Sigma = model.Sigma;
    w = model.weight;

    n = size(X,2);
    k = size(mu,2);
    logRho = zeros(n,k);

    for i = 1:k
        logRho(:,i) = loggausspdf(X,mu(:,i),Sigma(:,:,i));
    end
    logRho = bsxfun(@plus,logRho,log(w));
    T = logsumexp(logRho,2);
    llh = sum(T)/n; % loglikelihood
    logR = bsxfun(@minus,logRho,T);
    R = exp(logR);
    R = bsxfun(@times, R, weights);
    if (size(R,2)<2)
        fprintf('NaN');
    end
end

function model = maximization(X, R)
    if (size(R,2)<2)
        fprintf('NaN');
    end
    [d,n] = size(X);
    k = size(R,2);
    nk = sum(R,1);
    w = nk/n;
    mu = bsxfun(@times, X*R, 1./nk);
    
    if (isnan(mu))
        fprintf('NaN');
    end
    Sigma = zeros(d,d,k);
    sqrtR = sqrt(R);
    for i = 1:k
        Xo = bsxfun(@minus,X,mu(:,i));
        Xo = bsxfun(@times,Xo,sqrtR(:,i)');
        Sigma(:,:,i) = Xo*Xo'/nk(i);
        Sigma(:,:,i) = Sigma(:,:,i)+eye(d)*(1e-6); % add a prior for numerical stability
    end
    model.mu = mu;
    model.Sigma = Sigma;
    model.weight = w;
end

function y = loggausspdf(X, mu, Sigma)
    d = size(X,1);
    X = bsxfun(@minus,X,mu);
    [U,p]= chol(Sigma);
    if p ~= 0
        error('ERROR: Sigma is not PD.');
    end
    Q = U'\X;
    q = dot(Q,Q,1);  % quadratic term (M distance)
    c = d*log(2*pi)+2*sum(log(diag(U)));   % normalization constant
    y = -(c+q)/2;
end
