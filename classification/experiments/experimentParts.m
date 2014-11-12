function recRate = experimentParts(dataset,nrClasses, conf, confPart)
% experimentParts runs the experiment and returns a recognition rate

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
    
    if nargin < 4
        confPart = struct([]);
    end

    setts = settings();
    resDir = [setts.outputdir '/' dataset '/gcpr_parts/' num2str(nrClasses) '/'];
    if ~exist(resDir,'dir')
        mkdir(resDir);
    end
    if ~exist(setts.cachedir,'dir')
        mkdir(setts.cachedir);
    end
    
    
    [features, time_features, config, confDiffString] = experimentGeneral_extractGlobalFeatures(dataset, nrClasses, resDir, conf);
    [partFeatures, time_partFeatures, configParts, confPartDiffString] = experimentGeneral_extractPartFeatures(dataset, nrClasses, resDir, confPart);
    
    
    
    [~, labels_train, ~, labels_test ] = getDataset(dataset,'imagenames',nrClasses);
    if strcmp(config.useFlipped,'yes') && ~strcmp(configParts.useFlipped,'yes')
		for pi = 1:length(partFeatures)
			partFeatures(pi).hists_train = [partFeatures(pi).hists_train;partFeatures(pi).hists_train];
		end
    elseif strcmp(config.useFlipped,'yes') && ~strcmp(configParts.useFlipped,'yes')
		for pi = 1:length(features)
			features(pi).hists_train = [features(pi).hists_train;features(pi).hists_train];
		end
    end
        
    if strcmp(config.useFlipped,'yes') || strcmp(configParts.useFlipped,'yes')
		labels_train = [labels_train; labels_train];
    end

    if ~isempty(features)
        if ~isfield(features,'name')
            features(1).name = '';
            features(1).vocabulary = []; 
        end
    end
    
    features = [features partFeatures];
    
    if ~strcmp(configParts.noisyTrainingParts,'no')
        labels_train = repmat(labels_train, configParts.noisyTrainingPartsFactor, 1);
    end
    
    recRates_parts = [];
    mAP_parts = [];
%     for fi = 1:length(features)
%         tid_part = tic;
%         [ recRates_parts(fi), ~, scores, ~] = liblinearTrainTest( features(fi).hists_train, labels_train, features(fi).hists_test, labels_test, config );
%         mAP_parts(fi) = mean(evalAP(scores, labels_test));
%         part_time = toc(tid_part);
%         fprintf('train part model %d/%d (%.2f)\n',fi,length(features),part_time )
%     end


    hists_train = cat(2,features.hists_train);
    hists_test = cat(2,features.hists_test);
%     hists_train = features.hists_train;
%     hists_test =features.hists_test;

%     [~, labels_train, ~, labels_test ] = getDataset(dataset,'imagenames',nrClasses);
    [ recRate, ~, scores, ~] = liblinearTrainTest( hists_train, labels_train, hists_test, labels_test, config );

%     splitIdxs = crossvalind(labels_train, 5);
%     scores_train_crossval = [];
%     for k=1:5
%         [ ~, ~, scores_k, liblinearmodel] = liblinearTrainTest(hists_train(splitIdxs~=k,:),labels_train(splitIdxs~=k),hists_train(splitIdxs==k,:),labels_train(splitIdxs==k),config);
%         scores_train_crossval(splitIdxs==k,:) = scores_k;
%     end
            
            
    time_features
    
    recRates_parts
    
    recRate
    mAP = mean(evalAP(scores, labels_test))

    clear features
    clear partFeatures
    clear hists_train
    clear hists_test
    
    file = [resDir mfilename confDiffString confPartDiffString '.mat'];
%     file = [resDir 'globalGeneral_w' num2str(config.numWords) '_dict' config.codebookClusterAlgorithm  ...
%         '_dictcomp' config.codebookCompression num2str(config.codebookCompressionSize) ...
%         '_descr' config.descriptor '_pca' config.usePCACompression num2str(config.PCACompressionSize)  '_parts' configParts.useParts ...
%         '_global' config.useGlobal ...
%         '_noisyTrainParts' configParts.noisyTrainingParts '_' num2str(configParts.noisyTrainingPartsSigma) '_' num2str(configParts.noisyTrainingPartsFactor) ...
%         '_flipped' config.useFlipped  '.mat'];
    save(file,'-v7.3');
    
    toc(gtid)
end