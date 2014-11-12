%
% Extract a single feature type for a part
% This is done by changing the preprocessing function, such that a certain part is masked in the image.
% Caching is handled in experimentGeneral_extractPartFeatures.m
%
function [ part_features ] = extractPartFeaturesRLPooling( images_train, parts_train, labels_train, images_test, parts_test, config, vocabs)

    part_names = {'back    ' 'beak    ' 'belly    ' 'breast    ' 'crown    ' 'forehead' 'left eye' 'left leg' 'left wing' 'nape    ' 'right eye' 'right leg' 'right wing' 'tail    ' 'throat    '};

    parts = [];
    for pi = intersect([1 2 3 4 5 6 10 14 15],config.partSelection)
        % add new part specification
        parts(end+1).name = [ part_names{pi} ' (' config.featureExtractFun ')'];
        fprintf('\nCompute features for single part %d (%s)\n\n',pi,parts(end).name);

        % inline function (see below)
        % that changes config.preprocessFunctionArgs and takes care of corresponding part pairs
        config = prepareConfigForParts(images_train, parts_train, pi, config);

        % check whether we have to learn a new codebook
        if strcmp(config.descriptor,'plain')
            vocabulary.map=[];
        else
            if isempty(vocabs)
                [vocabulary] = vlfeatCreateCodebookAib(images_train, labels_train, config);
            else
                vocabulary = vocabs(length(parts)+1);
            end
        end
        % get the training features, standard BoW or Vlad, depending on the config
        % (the tag Vlad in the config does not matter at all!)
        [hists_train] = vlfeatFeatureExtractionVlad(images_train, vocabulary, config);

        if strcmp(config.useFlipped,'yes')
            flipConfig = config;
            flipConfig.preprocessing_flipImage = 1;
            [hists_train_flipped] = vlfeatFeatureExtractionVlad(images_train, vocabulary, flipConfig);                
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
        % get the testing features
        config = prepareConfigForParts(images_test, parts_test, pi, config);
        [hists_test] = vlfeatFeatureExtractionVlad(images_test, vocabulary, config);
        
        % add the features
        parts(end).hists_train = hists_train;
        parts(end).hists_test = hists_test;
        parts(end).vocabulary = vocabulary;
    end
    
    % 'left eye' 'left leg' 'left wing' 
    % TODO
    for pi = intersect([7 8 9],config.partSelection)
        parts(end+1).name = [ part_names{pi} ' and ' part_names{pi+4} ' (' config.featureExtractFun ')' ];
        fprintf('\nCompute features for single part %d (%s)\n\n',pi,parts(end).name);

        config = prepareConfigForParts(images_train, parts_train, pi, config);

        if strcmp(config.descriptor,'plain')
            vocabulary.map=[];
        else
            if isempty(vocabs)
                [vocabulary] = vlfeatCreateCodebookAib(images_train, labels_train, config);
            else
                vocabulary = vocabs(length(parts)+1);
            end
        end

        [hists_train] = vlfeatFeatureExtractionVlad(images_train, vocabulary, config);
        if strcmp(config.useFlipped,'yes')
            flipConfig = config;
            flipConfig.preprocessing_flipImage = 1;
            [hists_train_flipped] = vlfeatFeatureExtractionVlad(images_train, vocabulary, flipConfig);                
            hists_train = [hists_train; hists_train_flipped];
        end

        hists_train2 = [];
        if (config.bothSymmetricParts)
            config = prepareConfigForParts(images_train, parts_train, pi+4, config);
            [hists_train2] = vlfeatFeatureExtractionVlad(images_train, vocabulary, config);
            if strcmp(config.useFlipped,'yes')
                flipConfig = config;
                flipConfig.preprocessing_flipImage = 1;
                [hists_train2_flipped] = vlfeatFeatureExtractionVlad(images_train, vocabulary, flipConfig);                
                hists_train2 = [hists_train2; hists_train2_flipped];
            end
        end

        config = prepareConfigForParts(images_test, parts_test, pi, config);
        [hists_test] = vlfeatFeatureExtractionVlad(images_test, vocabulary, config);

        hists_test2 = [];
        if (config.bothSymmetricParts)
            config = prepareConfigForParts(images_test, parts_test, pi+4, config);
            [hists_test2] = vlfeatFeatureExtractionVlad(images_test, vocabulary, config);
        end
        
        if (max(max(abs(hists_test)))==0)
            hists_test = hists_test2;
        end
        if (max(max(abs(hists_train)))==0)
            hists_train = hists_train2;
        end

        parts(end).hists_train = hists_train;
        parts(end).hists_test = hists_test;
        parts(end).vocabulary = vocabulary;

    end
    part_features = parts;
end

%
%
%
%
%
function config = prepareConfigForParts(images, parts, pi, config)

    % get the proper part indices with left/right version?
    parts_idxs = (pi*2-1):(pi*2);
    
    % should we rotate parts? TODO
    if strcmp(config.rotateParts,'yes')
        config.preprocessFunction = @preprocessExtractPatchAtPositionAndRotate;
        part_pairs =  [10     6     4     3     6     5     9    12     7     5    13     8    11    12     6];
    else
        config.preprocessFunction = @preprocessExtractPatchAtPosition;
        part_pairs = 1:15;
    end
    
    % get the corresponding part partner
    pi2 = part_pairs(pi);
    parts2_idxs = (pi2*2-1):(pi2*2);

    % set the pre-processing method that takes care of the cropping during feature extraction
    args = {};
    args{1} = images;
    args{2} = [ parts(:,parts_idxs) repmat(config.preprocessing_relativePartSize,size(parts,1),1)];
    args{3} = parts(:,parts2_idxs);
    config.preprocessFunctionArgs = args;
end

%
% preprocessing method to crop a part out of the image
%
%
function [im, bbox] = preprocessExtractPatchAtPosition(im,bbox,imagename,config)

    currentIdx = find(strcmp(config.preprocessFunctionArgs{1},imagename));
    
    params = config.preprocessFunctionArgs{2}(currentIdx,:);
    
    x = params(1);
    y = params(2);
    
%     if isfield(config,'preprocessing_flipImage') && config.preprocessing_flipImage == 1 && x ~= -1
%         x = size(im,1) - x;
%     end
    
    if x==-1 && y==-1 % part not present
        im = imcrop(im,[1,1,-1,-1]); % return an empty image
        bbox.left = 1;
        bbox.right = 1;
        bbox.top = 1;
        bbox.bottom = 1;
    else
        bbox_width = bbox.right - bbox.left;
        bbox_height = bbox.bottom - bbox.top;
        if (config.preprocessing_cropToBoundingbox)
            width = sqrt(bbox_width*bbox_height*params(3)); % in percent, to account for variations in image size
            % param(3) encodes percentage of the area a single part occupies
        else
            width = sqrt(size(im,1)*size(im,2)*params(3));
        end
        height = width;
        xmin = x-width/2;
        ymin = y-width/2;

        im = imcrop(im,[xmin, ymin, width, height]);
        bbox.left = 1;
        bbox.right = size(im,2);
        bbox.top = 1;
        bbox.bottom = size(im,1);
    end
end

%
% preprocessing method to crop a part out of the image and perform some rotation
% using whatever position information
% TODO is this just rotating the bounding box?
function [im, bbox] = preprocessExtractPatchAtPositionAndRotate(im,bbox,imagename,config)

    currentIdx = find(strcmp(config.preprocessFunctionArgs{1},imagename));
    
    params = config.preprocessFunctionArgs{2}(currentIdx,:);
    
    x = params(1);
    y = params(2);
    
    params2 = config.preprocessFunctionArgs{3}(currentIdx,:);
    x2 = params2(1);
    y2 = params2(2);
    
    % TODO: what is this angle about?
    theta = atan2(x-x2,y-y2);
    
%     if isfield(config,'preprocessing_flipImage') && config.preprocessing_flipImage == 1 && x ~= -1
%         x = size(im,1) - x;
%     end
    
    if x==-1 && y==-1 % part not present
        im = imcrop(im,[1,1,-1,-1]); % return an empty image
        bbox.left = 1;
        bbox.right = 1;
        bbox.top = 1;
        bbox.bottom = 1;
    else
        bbox_width = bbox.right - bbox.left;
        bbox_height = bbox.bottom - bbox.top;

        width = sqrt(bbox_width*bbox_height*params(3)); % in percent, to account for variations in image size
        % param(3) encodes percentage of the area a single part occupies

        width = width;
        height = width;
        if x2==-1 && y2 == -1
            xmin = x-width/2;
            ymin = y-width/2;
            im = imcrop(im,[xmin, ymin, width, height]);
        else
            % where the hack is rotcrop defined?
            % TODO
            im = rotcrop(im,theta,[x y],[width width]);
        end
        bbox.left = 1;
        bbox.right = size(im,2);
        bbox.top = 1;
        bbox.bottom = size(im,1);
    end
end
