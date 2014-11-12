%experimentGeneral_extractPartFeatures Manages extraction of part features
%   This function is responsible for computing the part features according
%   to the supplied config. It returns the features, the time it took to
%   extract the features, the complete config-structure used and a string
%   containing the differences between the supplied and the default config
%   structure. 
%   If 'vocab' is not supplied, a vocabulary is computed using the training
%   data.
function [ partFeatures, time_partFeatures, configParts, confPartDiffString ] = experimentGeneral_extractPartFeatures( dataset, nrClasses, resDir, confPart, vocabs )

    % check whether a vocabulary was given
    if nargin < 5
        vocabs = [];
    end

    % Part selection
    configDefault.partSelection = 1:15;
    % Whether to extract left and right or just left of a part
    configDefault.bothSymmetricParts = 1;
    % Use GT or estimation for part locations
    configDefault.trainPartLocation='est';
    % Decaf Config
    % relu7, fc7_neuron_cudanet_out
    configDefault.decaf_layer='fc7_neuron_cudanet_out';
    
    % Default config for part extraction
    % Basically, we are again using a ColorName and SIFT BoW
    configDefault.pyramid_levels = 3; 
    configDefault.numWords = 100;
    configDefault.numSpatialX = [1 2 4]; 
    configDefault.numSpatialY = [1 2 4];
    configDefault.vocabSubset = 50;
    configDefault.colornameFeature_step = 3;
    configDefault.colornameFeature_width = 3;
    % configDefault.featureExtractFuns = {{'extractSiftFeatures'},{'extractColornameFeatures'}};
    configDefault.featureExtractFuns = {{'extractDecafFeatures'}};
    % why is this necessary here?
    configDefault.svm_C = 10;
    configDefault.svm_Kernel = 'linear';
    configDefault.homkermap_n = 1;
    configDefault.homkermap_gamma = 0.5;
    configDefault.preprocessing_useMask = 'grabcut';
    configDefault.preprocessing_cropToBoundingbox = 1;
    configDefault.preprocessing_boundingboxEnlargement = 1;
    configDefault.preprocessing_standardizeImage = true;
    configDefault.preprocessing_standardizeImageSize = 400;

    configDefault.codebookClusterAlgorithm = 'yael_kmeans';
%     configDefault.codebookClusterAlgorithm = 'kmeans';
%     configDefault.codebookClusterAlgorithm = 'random';

    configDefault.codebookCompression = 'none';
    configDefault.codebookCompressionSize = 10;
    configDefault.descriptor = 'plain'; %'histogram';
    %configDefault.preprocessFunction = @preprocessExtractPatch;
%     configDefault.usePCACompression = 'yes';
    configDefault.usePCACompression = 'yes';
    configDefault.PCACompressionSize = 2000;
    configDefault.useGlobal = 'yes';
    configDefault.useFlipped = 'yes';
    configDefault.useExternalVocabulary = 'no';
    configDefault.useRootSIFT = 'no';

    

    configPartsDefault = configDefault;
    configPartsDefault.numSpatialX = [1];
    configPartsDefault.numSpatialY = [1];
    configPartsDefault.pyramid_levels = 1; 
    
    % this parameter specifies the size of a part
    configPartsDefault.preprocessing_relativePartSize = 1/16;
    configPartsDefault.preprocessing_useMask = 'none';
    configPartsDefault.preprocessing_cropToBoundingbox = 1;
    configPartsDefault.preprocessing_standardizeImage = 0;
    configPartsDefault.numWords = min(100,configPartsDefault.numWords);
    configPartsDefault.codebookCompressionSize = min(100,configPartsDefault.codebookCompressionSize);

    % as a default, we are using ground-truth parts!
%    configPartsDefault.useParts = 'gt';
%     configPartsDefault.useParts = 'none';
     configPartsDefault.useParts = 'deepLearning';
%     configPartsDefault.useParts = 'dpm';
%     configPartsDefault.useParts = 'sdpm'; % not implemented yet


    % test with noisy part detections
    configPartsDefault.noisyTrainingParts = 'no';
%     configPartsDefault.noisyTrainingParts = 'yes';
%     configPartsDefault.noisyTrainingParts = 'additionally';
    configPartsDefault.noisyTrainingPartsSigma = 8;
    configPartsDefault.noisyTrainingPartsFactor = 3;

    configPartsDefault.useFlipped = 'yes';
    % Use the k'th nearest neighbour.
    % Important: this is not the k-nn method!!
  	configPartsDefault.k = 1;
    configPartsDefault.partEstimation_distanceMeasure = 'euclidean';
    configPartsDefault.rotateParts = 'no';
    
    [configParts,confPartDiffString] = parseConfigs(configPartsDefault, confPart);

    % In case, we want to use NN part estimation, we do that directly in
    % the beginning? TODO: no idea why, because this is called later on anyway
%     if strcmp(configParts.useParts,'nn')
%       [~,~,confDiffString] = partEstimationNN(dataset, nrClasses, resDir, confPart);
%       confPartDiffString = [confPartDiffString confDiffString];
%     end


    partFeatures = [];
    time_partFeatures = -1;
    
    if ~strcmp(configParts.useParts,'none')
        featureCacheFile = [resDir 'cache_partGeneralFeatures' confPartDiffString '.mat']; 

        % rather verbose filename
%         featureCacheFile = [resDir 'cache_partGeneralFeatures_w' num2str(configParts.numWords) '_dict' configParts.codebookClusterAlgorithm ... 
%             '_dictcomp' configParts.codebookCompression num2str(configParts.codebookCompressionSize) '_descr' configParts.descriptor  ...
%             '_pca' config.usePCACompression num2str(config.PCACompressionSize) '_parts' configParts.useParts ...
%             '_noisyTrainParts' configParts.noisyTrainingParts '_' num2str(configParts.noisyTrainingPartsSigma) '_' num2str(configParts.noisyTrainingPartsFactor) '.mat'];


        if exist(featureCacheFile,'file')
            fprintf('Loading part features from %s\n', featureCacheFile);
            load(featureCacheFile,'partFeatures','configParts','time_partFeatures');
        else
            fprintf('%s does not exist. Start computing features.\n',featureCacheFile);
            tic;
            if strcmp(configParts.useParts,'nn')
                if strcmp(dataset, 'cub200_2010_2011')
                	  error('Not implemented, because of missing parts');
                	
                    parts_cub200 = load([getenv('HOME') '/prom/results/finegrained/' 'cub200' '/' num2str(nrClasses) '/partsEstimatedUsingNN.mat'],'parts_test_estimated','parts_train_estimated');
                    parts_cub200_2011 = load([getenv('HOME') '/prom/results/finegrained/' 'cub200_2011' '/' num2str(nrClasses) '/partsEstimatedUsingNN.mat'],'parts_test_estimated','parts_train_estimated');
                    
                    parts_train_estimated = [parts_cub200_2011.parts_train_estimated; parts_cub200.parts_train_estimated];
                    parts_test_estimated  = [parts_cub200_2011.parts_test_estimated;  parts_cub200.parts_test_estimated ];
                else
                    %
                    % Use NN-matching to get the part positions 
                    %
                    [parts_test_estimated,parts_train_estimated,~] = partEstimationNN(dataset, nrClasses, resDir, confPart);
                end
            elseif strcmp(configParts.useParts,'deepLearning')
                [parts_test_estimated,parts_train_estimated,~] = partEstimationDeepLearning(dataset, nrClasses, resDir, confPart);
            elseif strcmp(configParts.useParts,'caffe')
                [parts_test_estimated,parts_train_estimated,~] = partEstimationCaffe(dataset, nrClasses, resDir, confPart);
            elseif strcmp(configParts.useParts,'gt')
                %
                % Get the real ground-truth part locations
                %
                [parts_train, ~, parts_test, ~] = getDataset(dataset, 'parts', nrClasses);
                parts_test_estimated = parts_test;
            elseif strcmp(configParts.useParts,'random')
                %
                % Use some random parts, interesting for debugging
                %
                [bboxes_train, labels_train, bboxes_test, labels_test] = getDataset('cub200_2011','bboxes',14);

                rand_parts_train = rand(size(parts_train));
                rand_parts_test = rand(size(parts_test));

                for ii=1:length(labels_train)
                    bbox = bboxes_train{ii};
                    width = bbox.right - bbox.left;
                    height = bbox.bottom - bbox.top;

                    rand_parts_train(ii,1:2:end) = rand_parts_train(ii,1:2:end)*width + bbox.left;
                    rand_parts_train(ii,2:2:end) = rand_parts_train(ii,2:2:end)*height + bbox.top;
                end

                for ii=1:length(labels_test)
                    bbox = bboxes_test{ii};
                    width = bbox.right - bbox.left;
                    height = bbox.bottom - bbox.top;

                    rand_parts_test(ii,1:2:end) = rand_parts_test(ii,1:2:end)*width + bbox.left;
                    rand_parts_test(ii,2:2:end) = rand_parts_test(ii,2:2:end)*height + bbox.top;
                end
                
                rand_parts_train = round(rand_parts_train);
                rand_parts_test = round(rand_parts_test);

                parts_test_estimated = rand_parts_test;
                
            elseif strcmp(configParts.useParts,'dpm') | strcmp(configParts.useParts,'ssdpm')
                if strcmp(configParts.useParts,'dpm')
                    %
                    % Perform DPM detection with some pre-obtained results
                    %
                    if nrClasses == 14
                        tmp = load('/home/goering/prom/results/old/cluster/finegrained/cub200_2011/felzenszwalbdeformablepartmodels_n3_14_parts2/partsBest.mat');
                    else
                        tmp = load('/home/goering/prom/results/old/cluster/finegrained/cub200_2011/felzenszwalbdeformablepartmodels_n5_200_parts/partsBest.mat');
                    end
                elseif strcmp(configParts.useParts,'ssdpm')
                    %
                    % Perform Supervised-DPM detection with some pre-obtained results
                    %

                    tmp = load([getenv('HOME') '/prom/results/finegrained/cub200_2011/' num2str(nrClasses) '/superviseddeformablepartmodels_own_colors_' num2str(nrClasses) '/partsBest.mat']);
                else
                    error('unknown option')
                end
                
                % now pre-process the detections
                parts_estimated_dpm = [];
                for ii = 1:length(tmp.part_overlap_detections)
                    currentDetection = tmp.part_overlap_detections(ii).parts;
                    currentDetection = flipud(currentDetection);
                    tmp2(currentDetection(:,2),:) = currentDetection;
                    currentDetection = tmp2;

                    x = mean(currentDetection(:,[4 6]),2);
                    y = mean(currentDetection(:,[5 7]),2);
                    
                    x(currentDetection(:,2)==0) = -1;
                    y(currentDetection(:,2)==0) = -1;
    
                    parts_estimated_dpm(ii,:) = reshape([x(2:end) y(2:end)]',1,30);
                end
                
                [images_train, ~, images_test, ~] = getDataset(dataset, 'imagenames', nrClasses);

                parts_train_estimated = parts_estimated_dpm( (1:length(parts_estimated_dpm)) <= length(images_train) ,:);
                parts_test_estimated = parts_estimated_dpm( (1:length(parts_estimated_dpm)) > length(images_train) ,:);

            else
            	error('not implemented');
            end
           
            % get the real part positions if available
            if strcmp(dataset,'cub200_2011')
                [parts_train, ~, parts_test, ~] = getDataset(dataset, 'parts', nrClasses);
            elseif strcmp(dataset,'cub200_2011_facingright')
                [parts_train, ~, parts_test, ~] = getDataset(dataset, 'parts', nrClasses);
            else
                parts_train = parts_train_estimated;
                parts_test = parts_test_estimated;
            end
            [images_train, labels_train, images_test, ~ ] = getDataset(dataset,'imagenames',nrClasses);

            %
            % add noise to training parts
            %
            if ~strcmp(configParts.noisyTrainingParts,'no')
                parts_train_noisy = [];
                for ii = 1:configParts.noisyTrainingPartsFactor
                    
                    if ii == 1 && strcmp(configParts.noisyTrainingParts,'additionally')
                        parts_train_noisy_cur = parts_train;
                    else
                        parts_train_noisy_cur = parts_train + randn(size(parts_train)) * configParts.noisyTrainingPartsSigma;
                        parts_train_noisy_cur(parts_train == -1) = -1;
                    end

                    parts_train_noisy = [parts_train_noisy; parts_train_noisy_cur];
                end
                parts_train = parts_train_noisy;
            end
            %%%%

            %
            % feature extraction with the given part locations
            %
            fprintf('Feature extraction with estimated/given part locations\n');
            if (strcmp(configParts.trainPartLocation,'est'))
                partFeatures = extractMultiplePartFeatures(images_train, labels_train, parts_train_estimated, images_test, parts_test_estimated, configParts, vocabs);
            else
                partFeatures = extractMultiplePartFeatures(images_train, labels_train, parts_train, images_test, parts_test_estimated, configParts, vocabs);
            end
            time_partFeatures = toc;

            save(featureCacheFile,'partFeatures','configParts','time_partFeatures','parts_train','parts_test_estimated','-v7.3');
        end
    end

end



    function res = compare(a,b)
        
        if ~strcmp(class(a),class(b)) 
            assert('different types...')
        end
        
        res = false;
        if ischar(a)
            res = strcmp(a,b);
        elseif isnumeric(a)
            if isscalar(a) && isscalar(b)
                res = a == b;
            else
                if all(size(a) == size(b))
                    res = all(a == b);
                else
                    res = false;
                end
            end
        elseif iscell(a)
            if all(size(a) == size(b))
                res = false;
            else
                partRes = [];
                for ii = 1:length(a)
                    partRes(ii) = compare(a{ii},b{ii});
                end
                res = all(partRes);
            end
        else
            assert(['type not implemented: ' class(a)])
        end
        
    end
