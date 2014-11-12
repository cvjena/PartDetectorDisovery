% Computes part features for several different feature types
% Calls extractPartFeaturesRLPooling for each individual feature type
% Features are extracted using each function from config.featureExtractFuns
function features = extractMultiplePartFeatures(images_train, labels_train, parts_train, images_test, parts_test, config, vocabs)

    if iscell(config.featureExtractFuns{1}) % hack to make the config stuff work
        config.featureExtractFuns = config.featureExtractFuns{1};
    end
    
    features = [];
    for ii = 1:length(config.featureExtractFuns)
        config.featureExtractFun = config.featureExtractFuns{ii};

        if isempty(vocabs)
            parts = extractPartFeaturesRLPooling( images_train, parts_train, labels_train, images_test, parts_test, config, []);
        else
            parts = extractPartFeaturesRLPooling( images_train, parts_train, labels_train, images_test, parts_test, config, vocabs(ceil(length(config.featureExtractFuns)*(1:length(vocabs))/length(vocabs)) == ii) );
        end
        
        fl = length(features);
        for jj = 1:length(parts)
            if length(features) == 0
                features = parts;
            else
                features(fl + jj) = parts(jj);
            end
        end
    end

end
