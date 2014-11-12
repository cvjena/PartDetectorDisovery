% experimentParts_knn uses the k best part estimations to compute the final
% classification result
function recRate = experimentParts_knn(dataset,nrClasses, conf, confPart)

    maxK = 6;

    gtid = tic;
    if nargin < 1
        dataset = 'cub200_2011';
    end
    
    if nargin < 2
        nrClasses = 14;
    end
    if nargin < 3
    	conf = struct([]);
    end

    % the standard is to use some parts
    %if nargin < 4 || ~isfield(confPart.useParts)
    warning('Running the KNN experiment without using parts does not make any sense, therefore all confPart.useParts options are set to nn\n');
    if nargin<4 || isempty(confPart)
      confPart = struct([]);
    end
    confPart(1).useParts = 'nn';
    %end


    % change the outputdir in the file settings.m
    setts = settings();

    % create the directory used to store the results,
    % in my case: /home/rodner/experiments/finegrained/results/cub200_2011/gcpr_parts/14
    resDir = [setts.outputdir '/' dataset '/gcpr_parts/' num2str(nrClasses) '/'];
    if ~exist(resDir,'dir')
        mkdir(resDir);
    end
    
    % extract global features for the whole dataset
    % standard global features are colorname and SIFT BoW histograms
    [features, time_features, config, confDiffString] = experimentGeneral_extractGlobalFeatures(dataset, nrClasses, resDir, conf);
    
    % extract part features using the k'th nearest neighbour
    % using k=1:1 does not make sense here!
    % but otherwise the features would be lost
    for k=1:1
        confPart(1).k = k;
        [partFeatures, ~, configParts, confPartDiffString] = experimentGeneral_extractPartFeatures(dataset, nrClasses, resDir, confPart);
    end
    
    [~, labels_train, ~, labels_test ] = getDataset(dataset,'imagenames',nrClasses);
    if strcmp(config.useFlipped,'yes') && ~strcmp(configParts.useFlipped,'yes')
        %
        % duplicate the part features for the flipped image
        % only in the case that we did not take care of flipping during extractPartFeatures
        % (because configParts.useFlipped) was off, but we have general flipping set to true
        %
        for pi = 1:length(partFeatures)
          partFeatures(pi).hists_train = [partFeatures(pi).hists_train;partFeatures(pi).hists_train];
        end
        elseif strcmp(config.useFlipped,'yes') && ~strcmp(configParts.useFlipped,'yes')
        for pi = 1:length(features)
          features(pi).hists_train = [features(pi).hists_train;features(pi).hists_train];
        end
    end
       
    % if we use any flipping, duplicate the labels
    if strcmp(config.useFlipped,'yes') || strcmp(configParts.useFlipped,'yes')
        labels_train = [labels_train; labels_train];
    end

    % TODO? Why should we have features already
    if ~isempty(features)
        if ~isfield(features,'name')
            features(1).name = '';
            features(1).vocabulary = []; 
        end
    end

    scores = {};
    % now we use several part estimation from several nearest neighbours and perform feature extraction
    % and classification!
    % TODO: no idea why
    % previous features seem to be ignored completely
    for k=1:maxK 
        fprintf('Compute scores for k=%d\n',k);
        tic;
        confPart(1).k=k;
        [partFeatures, ~, ~, ~] = experimentGeneral_extractPartFeatures(dataset, nrClasses, resDir, confPart);
        
        % partFeatures is a 24 field of structs with hists_train, etc.
        % Show some values of the feature matrix with: partFeatures(1).hists_test(1:10, 1:10)

        % This looks like a odd piece of code, combining fields of structs
        features_com = [features partFeatures];
        hists_train = cat(2,features_com.hists_train);
        hists_test = cat(2,features_com.hists_test);
        [ recRate(k), ~, scores{k}, ~] = liblinearTrainTest( hists_train, labels_train, hists_test, labels_test, config );
        disp( scores{k}(1:10) )

       	[~,labels_est{k}]=max(scores{k});
       	labels_correct{k} = labels_est{k} == labels_test';
        toc
    end

    % TODO: ?
    if ~strcmp(configParts.noisyTrainingParts,'no')
        labels_train = repmat(labels_train, configParts.noisyTrainingPartsFactor, 1);
    end

%% Some old code
%
%    recRates_parts = [];
%    mAP_parts = [];
%    for fi = 1:length(features)
%        tid_part = tic;
%        [ recRates_parts(fi), ~, scores, ~] = liblinearTrainTest( features(fi).hists_train, labels_train, features(fi).hists_test, labels_test, config );
%        mAP_parts(fi) = mean(evalAP(scores, labels_test));
%        part_time = toc(tid_part);
%        fprintf('train part model %d/%d (%.2f)\n',fi,length(features),part_time )
%    end
%    hists_train = cat(2,features.hists_train);
%    hists_test = cat(2,features.hists_test);
%     [~, labels_train, ~, labels_test ] = getDataset(dataset,'imagenames',nrClasses);
 

    % labels_correct{k} are the indicators when using the kth nearest neighbour
    % cat(1,labels_correct{:}) is the simple vertical concatenation of all the vectors
    % labels_correct_all is adding the two vectors together
    labels_correct_all = sum(cat(1,labels_correct{:}));
    % recRate_all is then the recognition rate we get, when we choose the optimal nearest neighbour among the k
    recRate_all = mean(labels_correct_all > 0);

    fprintf('Oracle recognition rate when the oracle can choose among all %d neighbours: %f\n', maxK, recRate_all);
   
    fprintf('Recognition rates when using the k-th nearest neighbour only: ');
    disp(recRate)
    
    for ii=1:length(labels_est)
        if ii==1
            recRate_vote(ii) = mean(labels_est{1}==labels_test');
            recRate_mean(ii) = mean(labels_est{1}==labels_test');
        else
            a=hist( cat(1,labels_est{1:ii}), 1:nrClasses);
            % maximum voting
            [~,b]=max(a);
            recRate_vote(ii) = mean(b==labels_test');
            
            % mean voting
            [~,b]=max(mean(cat(3,scores{1:ii}),3));
            recRate_mean(ii) = mean(b==labels_test');
        end
    end
  
    % some visualization plot
    %title('Average voting vs. Maximum voting');
    %plot([recRate_mean;recRate_vote]')
    
    recRate_vote
    recRate_mean

    % proper decisions are:
    recRate = recRate_mean;
    
%    mAP = mean(evalAP(scores, labels_test))

    clear features
    clear partFeatures
    clear hists_train
    clear hists_test
    
    file = [resDir mfilename confDiffString confPartDiffString '.mat'];
    fprintf('saving all the results to: %s', file);
    save(file,'-v7.3');

%     file = [resDir 'globalGeneral_w' num2str(config.numWords) '_dict' config.codebookClusterAlgorithm  ...
%         '_dictcomp' config.codebookCompression num2str(config.codebookCompressionSize) ...
%         '_descr' config.descriptor '_pca' config.usePCACompression num2str(config.PCACompressionSize)  '_parts' configParts.useParts ...
%         '_global' config.useGlobal ...
%         '_noisyTrainParts' configParts.noisyTrainingParts '_' num2str(configParts.noisyTrainingPartsSigma) '_' num2str(configParts.noisyTrainingPartsFactor) ...
%         '_flipped' config.useFlipped  '.mat'];

    
    toc(gtid)
end
