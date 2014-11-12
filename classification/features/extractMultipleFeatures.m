% extractMultipleFeatures extracts features for each of the functions given
% in config.featureExtractFuns
%
% The function is called by experimentGeneral_extractGlobalFeatures, which takes care of the default config
% and the caching.
function features = extractMultipleFeatures(dataset, nrClasses, config, vocabs)

    [images_train, labels_train, images_test, ~ ] = getDataset(dataset,'imagenames',nrClasses);

    % hack to make the config stuff work
    if iscell(config.featureExtractFuns{1}) 
        config.featureExtractFuns = config.featureExtractFuns{1};
    end
   
    % Go through all feature methods
    % Example conf: configDefault.featureExtractFuns = {{'extractSiftFeatures','extractColornameFeatures'}};
    for ii = 1:length(config.featureExtractFuns)
        config.featureExtractFun = config.featureExtractFuns{ii};
    
        % if no predefined BoW codebook/vocabulary is given, we have to run the clustering again
        if strcmp(config.descriptor,'plain')
            vocabulary.map=[];
        else
            if isempty(vocabs)
                [vocabulary] = vlfeatCreateCodebookAib(images_train, labels_train, config);
            else
                vocabulary = vocabs(ii);
            end
        end
        % first, don't flip the image
        config.preprocessing_flipImage = 0;
        % perform pooling and coding with VLAD
        % http://www.vlfeat.org/sandbox/api/vlad-fundamentals.html
        [hists_train] = vlfeatFeatureExtractionVlad(images_train, vocabulary, config);
        
        % now flip the image in case
        if strcmp(config.useFlipped,'yes')
            % also compute features of the flipped version of the images
            % and append them to the results
	        config.preprocessing_flipImage = 1;
    	    [hists_train_flipped] = vlfeatFeatureExtractionVlad(images_train, vocabulary, config);
          hists_train = [hists_train; hists_train_flipped];
        end

        disp('==============\nTesting data\n');
        disp('==============\nTesting data\n');
        disp('==============\nTesting data\n');
        disp('==============\nTesting data\n');
        disp('==============\nTesting data\n');
        disp('==============\nTesting data\n');
        disp('==============\nTesting data\n');
        disp('==============\nTesting data\n');
        disp('==============\nTesting data\n');
        disp('==============\nTesting data\n');
        disp('==============\nTesting data\n');
        % be sure to change the flipping flag
        config.preprocessing_flipImage = 0;
        [hists_test] = vlfeatFeatureExtractionVlad(images_test, vocabulary, config);
            
        % save the features with a proper name
        features(ii).hists_train = hists_train;
        features(ii).hists_test = hists_test;
        features(ii).vocabulary = vocabulary;
        features(ii).name = ['global ' config.featureExtractFuns{ii}];
    end

end
