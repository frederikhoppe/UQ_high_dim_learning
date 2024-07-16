designmode = 'fourier'; %gaussian_complex

p = 1000;
s = ceil(0.05*p);
n = ceil(0.4*p);
alpha = 0.05;

iterations = 500;
statdata = cell(1,iterations);

lasso2norm = zeros(iterations,1);
lassoinfnorm = zeros(iterations,1);
restinfnorm = zeros(iterations,1);
rest2norm = zeros(iterations,1);
gaussinfnorm = zeros(iterations,1);
gauss2norm = zeros(iterations,1);
deblassoinfnorm = zeros(iterations,1);
deblasso2norm = zeros(iterations,1);

%% initialize design matrix
if strcmp(designmode,'fourier')
    rows = transpose(randperm(p,n));
    Sub = linop_subsample([n,p], rows); %subsampling operator
    FFT = linop_fft( p, p, 'c2c' ); %without 1/sqrt(p);
    C = linop_compose( Sub, FFT ); %composed operator with quadratic (pxp) fft
    scale = linop_scale(1/sqrt(n)); %technical steps for TFOCS
    Crip = linop_compose(scale, C);
    ownop = @(x,mode)ownlinop(n,p,Crip, FFT, rows, x,mode);
else
    designsub = complex(randn(n,p),randn(n,p))./sqrt(2); %standard gaussian
    ripdesign = designsub./sqrt(n); %normalize matrix in order to satiesfy RIP
    designoplasso = linop_matrix(ripdesign, 'C2C'); %technical step for TFOCS
end

if ~strcmp(designmode,'fourier') %for Fourier diagonal entries are one
        samplecovariance = ctranspose(designsub)*designsub./n;
end

sigma = 0.15; %create noise
sigmahat = sigma; %for simplicity, alternative:scaled LASSO
        
%% choose regularization parameter
lambda = 2*sigmahat/n*(2+sqrt(12*log(p))); %factor 25 due to cross validation
%the divisor n instead of sqrt(n) comes from the normalization of TFOCS lasso
%implementation

%% construct confidence interval
phiinvers = sqrt(2*log(1/alpha)); %alpha/2 quantile complex noise

if strcmp(designmode,'fourier')
    radius = phiinvers*sigmahat/sqrt(2*n)*ones(p,1);
else
    radius = phiinvers*sigmahat/sqrt(2*n)*sqrt(diag(samplecovariance));
end


%parfor j=1:iterations %parallel
for j=1:iterations
    %% initialize groundtruth
    T = randperm(p,s); %set fix active set of groundtruth
    x0 = zeros(p,1);
    x0(T) = complex(randn(s,1),randn(s,1))./sqrt(2); %set normal distributed entries of gt
    x0 = x0./norm(x0,2);
 
    %% initialize measurement
    if strcmp(designmode,'fourier')
        ywithoutnoise = C(x0, 1); %compute measurement
    else
        ywithoutnoise = designsub*x0;
    end

    %% add noise
    z = sigma/sqrt(2).*(randn(n,1)+1i*randn(n,1)); %complex Gaussian noise
    y = ywithoutnoise+z; %measurements with noise
    relativenoiselevel = norm(z,2)/norm(ywithoutnoise,2); %noise level relative to measurements

    %% solving LASSO
    if strcmp(designmode,'fourier') 
        beta = solver_L1RLS(ownop, y./sqrt(n), lambda); %solve LASSO in Fourier case
    else
        beta = solver_L1RLS(designoplasso, y./sqrt(n), lambda); %solve LASSO in Gaussian case
    end
    
    %% debiasing step
    if strcmp(designmode,'fourier')
        residual = y-C(beta,1);
        yinit = zeros(p,1); %apply adjoint subsampled Fourier
        yinit(rows) = residual; %apply adjoint subsampled Fourier
        dsl = beta+FFT(yinit,2)./n; %debiased LASSO
    else
        residual = y-designsub*beta;
        dsl = beta+ctranspose(designsub)*residual./n; %debiased LASSO
    end
        
    %% error analysis
    if ~strcmp(designmode,'fourier')
        ownop = designoplasso;
    end
    lassodifference = beta-x0;
    lasso2norm(j) = norm(lassodifference,2);     
    lassoinfnorm(j) = norm(lassodifference, inf);
    step = ownop(lassodifference,1);
    restterm = ownop(step,2) - lassodifference;
    restinfnorm(j) = norm(restterm, inf);
    rest2norm(j) = norm(restterm, 2);
    gaussterm = ownop(z,2)/sqrt(n);
    gaussinfnorm(j) = norm(gaussterm, inf);
    gauss2norm(j) = norm(gaussterm, 2);
    testterm = gaussterm - restterm;
    udifference = dsl-x0;
    deblassoinfnorm(j) = norm(udifference, inf);
    deblasso2norm(j) = norm(udifference, 2);
    norm(testterm-udifference,inf)
        
    statdata{j} = restterm;
    Rmatrix(:,j) = restterm;
end

avlasso2norm = norm(lasso2norm,1)/iterations;
avlassoinfnorm = norm(lassoinfnorm,1)/iterations;
avrestinfnorm = norm(restinfnorm,1)/iterations;
avrest2norm = norm(rest2norm,1)/iterations;
avgaussinfnorm = norm(gaussinfnorm,1)/iterations;
avgauss2norm = norm(gauss2norm,1)/iterations;
avdeblassoinfnorm = norm(deblassoinfnorm,1)/iterations;
avdeblasso2norm = norm(deblasso2norm,1)/iterations;


%% estimating data-driven radii
zdaten = zeros(p*iterations,1);
for i=1:iterations
    zdaten((i-1)*p+1:i*p) = statdata{1,i};
end
estMeanR = 1/(iterations*p)*norm(zdaten,1);
estVarR = 1/(iterations*p-1) * sum((abs(zdaten) - estMeanR*ones(iterations*p,1)).^2);

for k=1:p
    estMeanR_pixel(k) = 1/(iterations)*norm(transpose(Rmatrix(k,:)),1);
    meanRealR(k) = 0; %1/(iterations)*sum(real(transpose(Rmatrix(k,:))));
    estVarR_pixel(k) = 1/(iterations-1) * sum((abs(transpose(Rmatrix(k,:))) - estMeanR_pixel(k)*ones(iterations,1)).^2);
    varianceRealR(k) = 1/(iterations-1) * sum((real(transpose(Rmatrix(k,:))) - meanRealR(k)*ones(iterations, 1)).^2);
end

if strcmp(designmode,'fourier')
    average_cov=1; 
else
    average_cov = sum(diag(samplecovariance).^(1/2))/p;
end

l=iterations*p;
% define radius as a function of gamma
gamma0 = 0.5;
rgamma = @(gamma) (sigma * average_cov / sqrt(n)) * sqrt(log(1 / (gamma * alpha))) + ...
    sqrt((l^2 - 1) / (l^2 * (1 - gamma) * alpha - l)) * sqrt(estVarR) + ...
    estMeanR;


% Define bounds for gamma
lb = 0;
ub = 1-1/(iterations*alpha);

% Define options for fmincon
options = optimoptions('fmincon', 'Display', 'iter', 'Algorithm', 'interior-point');

% Use fmincon to minimize f(gamma) with constraints
[gamma_min, fval, exitflag] = fmincon(rgamma, gamma0, [], [], [], [], lb, ub, [], options);

% Display the result
if exitflag > 0
    fprintf('The value of gamma that minimizes the function is: %.4f\n', gamma_min);
    fprintf('The minimum value of the function is: %.4f\n', fval);
else
    fprintf('fmincon did not converge to a solution.\n');
end

c_l = sqrt( ((iterations)^2-1) / ((iterations)^2*(1-gamma_min)*alpha-iterations) );

for k=1:p

    if strcmp(designmode,'fourier')
        radiusnew(k) = sigma/sqrt(n)*sqrt(log(1/(gamma_min*alpha)))...
        + c_l*sqrt(estVarR_pixel(k)) + estMeanR_pixel(k);
    else
        radiusnew(k) = sigma*sqrt(samplecovariance(k,k))/sqrt(n)*sqrt(log(1/(gamma_min*alpha)))...
        + c_l*sqrt(estVarR_pixel(k)) + estMeanR_pixel(k);
    end
end


%% Estimating Gaussian adjusted radii
if strcmp(designmode, 'fourier')
    for k=1:p
        radiusGaussR(k) = phiinvers*sqrt(sigmahat^2 + 2*n*varianceRealR(k))/sqrt(2*n); %factor 2 since sigmaestimate is for real part
    end
else
    for k=1:p
        radiusGaussR(k) = phiinvers/sqrt(2*n)*sqrt(sigmahat^2*diag(samplecovariance(k,k)) + 2*n*varianceRealR(k));
    end
end

iterEv = 250;
hitrate = zeros(p,1);
hitrateGauss = zeros(p,1);
hitratenew = zeros(p,1);
hitratesupport = zeros(iterEv,1);
hitratesupportnew = zeros(iterEv,1);
hitratesupportGauss = zeros(iterEv,1);
Cov = zeros(p,1);
Covs = zeros(iterEv,1);
Covnew = zeros(p,1);
CovSnew = zeros(iterEv,1);
CovGauss = zeros(p,1);
CovSGauss = zeros(iterEv,1);
    
for j=1:iterEv
%% initialize further groundtruth
    T = randperm(p,s); %set fix active set of groundtruth
    x0 = zeros(p,1);
    x0(T) = complex(randn(s,1),randn(s,1))./sqrt(2); %set normal distributed entries of gt
    x0 = x0./norm(x0,2);

    %% initialize measurement
    if strcmp(designmode,'fourier')
        ywithoutnoise = C(x0, 1); %compute measurement
    else
        ywithoutnoise = designsub*x0;
    end

    %% add noise
    z = sigma/sqrt(2).*(randn(n,1)+1i*randn(n,1)); %complex Gaussian noise
    y = ywithoutnoise+z; %measurements with noise    

    %% solving LASSO
    if strcmp(designmode,'fourier') 
        beta = solver_L1RLS(ownop, y./sqrt(n), lambda); %solve LASSO in Fourier case
    else
        beta = solver_L1RLS(designoplasso, y./sqrt(n), lambda); %solve LASSO in Gaussian case
    end
    
    %% debiasing step
    if strcmp(designmode,'fourier')
        residual = y-C(beta,1);
        yinit = zeros(p,1); %apply adjoint subsampled Fourier
        yinit(rows) = residual; %apply adjoint subsampled Fourier
        dsl = beta+FFT(yinit,2)./n; %debiased LASSO
    else
        residual = y-designsub*beta;
        dsl = beta+ctranspose(designsub)*residual./n; %debiased LASSO
    end

    %% Evaluate radius
    for l=1:p
        if abs(x0(l)-dsl(l)) < radius(l)
            hitrate(l) = hitrate(l)+1;  %over samples
        end
    end
    for l=1:s
        if abs(x0(T(l))-dsl(T(l))) < radius(T(l))
            hitratesupport(j) = hitratesupport(j)+1;    %within one sample
        end
    end
    Covs(j) = norm(hitratesupport(j),1)/s;
    %% Evaluate new radius
    for l=1:p
        if abs(x0(l)-dsl(l)) < radiusnew(l)
            hitratenew(l) = hitratenew(l)+1;    %over samples
        end
    end
    for l=1:s
        if abs(x0(T(l))-dsl(T(l))) < radiusnew(T(l))   
            hitratesupportnew(j) = hitratesupportnew(j)+1;  %within one sample
        end
    end
    CovSnew(j) = norm(hitratesupportnew(j),1)/s;
    
    for l=1:p
        if abs(x0(l)-dsl(l)) < radiusGaussR(l)
            hitrateGauss(l) = hitrateGauss(l)+1;
        end
    end
    
    for l=1:s
        if abs(x0(T(l))-dsl(T(l))) < radiusGaussR(T(l))
            hitratesupportGauss(j) = hitratesupportGauss(j)+1;
        end
    end
    CovSGauss(j) = norm(hitratesupportGauss(j),1)/s;


end
Covnew = hitratenew./iterEv;
Cov = hitrate./iterEv;
CovGauss = hitrateGauss/iterEv;

avhitrate = norm(Cov,1)/p;
avhitratesupport = norm(Covs,1)/iterEv;
avhitratenew = norm(Covnew,1)/p;
avhitratesupportnew = norm(CovSnew,1)/iterEv;
avhitrateGauss = norm(CovGauss,1)/p;
avhitratesupportGauss = norm(CovSGauss,1)/iterEv;




function y = ownlinop(n, p, C, FFT, rows, x, mode)
switch mode
    case 0, y = [n,p];
    case 1, y = C(x,1);
    case 2, yinit = zeros(p,1);
        yinit(rows) = x;
        y = sqrt(1/n).*FFT(yinit,2);
end
end