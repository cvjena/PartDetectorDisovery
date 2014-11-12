function [features, time_features, config, confDiffString] = experimentGeneral_extractGlobalFeatures(dataset,nrClasses, resDir, conf, vocabs)
    % This function prepares the config and performs the caching for 
    % finally computing global features

    % Is there a given vocabulary already?
    % In general, vocab is a set of vocabularies for each config.featureExtractFuns,
    % e.g., one for extractSiftFeatures and extractColornameFeatures
    if nargin < 5
        vocabs = [];
    end

    % Decaf Config
    % relu7
    % fc7_neuron_cudanet_out
    configDefault.decaf_layer='fc7_neuron_cudanet_out';
    
    % number of pyramids for spatial pyramid matching
    configDefault.pyramid_levels = 3; 

    % Bag-of-words vocabulary size
    configDefault.numWords = 100;
    configDefault.numSpatialX = [1 2 4]; 
    configDefault.numSpatialY = [1 2 4];
    configDefault.vocabSubset = 50;
    configDefault.colornameFeature_step = 3;
    configDefault.colornameFeature_width = 3;
    configDefault.featureExtractFuns = {{'extractDecafFeatures'}};
    configDefault.svm_C = 10;
    configDefault.svm_Kernel = 'linear';
    configDefault.homkermap_n = 1;
    configDefault.homkermap_gamma = 0.5;

    % standard is to use a GrabCut segmentation,
    % other options are the ground-truth bounding box and no masking at all
    configDefault.preprocessing_useMask = 'grabcut'; % grabcut, bbox, none
    configDefault.preprocessing_cropToBoundingbox = 1;
    configDefault.preprocessing_boundingboxEnlargement = 0;

    % resize the image 
    configDefault.preprocessing_standardizeImage = true;
    configDefault.preprocessing_standardizeImageSize = 400;

    % possible bag-of-words cluster algorithms
    configDefault.codebookClusterAlgorithm = 'yael_kmeans';
%     configDefault.codebookClusterAlgorithm = 'kmeans';
%     configDefault.codebookClusterAlgorithm = 'random';

    % FIXME: codebook compression?
    configDefault.codebookCompression = 'none';
    configDefault.codebookCompressionSize = 10;

    configDefault.descriptor = 'plain'; %'histogram';
    %configDefault.preprocessFunction = @preprocessExtractPatch;

    % some PCA preprocessing
    configDefault.usePCACompression = 'yes';
    %configDefault.usePCACompression = 'yes';
    configDefault.PCACompressionSize = 2000;

    % the standard is to use global features :)
    configDefault.useGlobal = 'yes';
    configDefault.useFlipped = 'yes';
    configDefault.useExternalVocabulary = 'no';
    configDefault.useRootSIFT = 'no';
   
    % merge default config with the given one
    [config,confDiffString] = parseConfigs(configDefault, conf);
    
    tic;
    
    features = [];
    time_features = -1;

    % check whether we really want to extract features
    if strcmp(config.useGlobal,'yes')
        % cache file
        featureCacheFile = [resDir 'cache_globalGeneralFeatures' confDiffString '.mat'];
        % very specific cache file name, not really necessary anymore
        % featureCacheFile = [resDir 'cache_globalGeneralFeatures_w' num2str(config.numWords) '_dict' config.codebookClusterAlgorithm  '_dictcomp' config.codebookCompression num2str(config.codebookCompressionSize) '_descr' config.descriptor '_pca' config.usePCACompression num2str(config.PCACompressionSize) '_flipped' config.useFlipped '.mat'];

        if exist(featureCacheFile,'file')
            fprintf('Loading global features from cache file: %s\n', featureCacheFile);
            load(featureCacheFile,'features','time_features');
        else
            fprintf('The cache file %s does not exist. Start computing features.\n',featureCacheFile);
            tic;
            features = extractMultipleFeatures(dataset, nrClasses, config, vocabs);
            time_features = toc;

            save(featureCacheFile,'features','config','time_features','-v7.3');
        end
    end
end

