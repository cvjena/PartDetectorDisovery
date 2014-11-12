function [model] = vlfeatCreateCodebookAib(images, labels, config)
%vlfeatCreateCodebookAib creates codebook according to the supplied config
%   struct

if nargin ~= 3
    error('not enough arguments')
end

conf.codebookClusterAlgorithm = 'kmeans';
conf.numWords = 500;
conf.quantizer = 'kdtree' ;
conf.phowOpts = {'Step', 3, 'Color','opponent'} ;
conf.randSeed = 1 ;
conf.preprocessing_useGrabcutMask = false;
conf.preprocessing_cropToBoundingbox = false;
conf.preprocessing_standardizeImage = true;
conf.preprocessing_standardizeImageSize = 480;
conf.vocabSubset = 30;
conf.descrsSubset = 10e4;
conf.basedir = [getenv('HOME') '/'];
conf.excludeGTBoundingBoxes = false;
conf.featureExtractFun = @extractSiftFeatures;

if exist('config','var')
    for field = fieldnames(config)'
        conf.(field{1}) = config.(field{1}); % why is {1} necessary?
    end
end

model.quantizer = conf.quantizer;

randn('state',conf.randSeed) ;
rand('state',conf.randSeed) ;
vl_twister('state',conf.randSeed) ;

% --------------------------------------------------------------------
%                                                     Train vocabulary
% --------------------------------------------------------------------

nrClasses = max(labels);

fprintf('Train vocabulary\n')


  % Get some PHOW descriptors to train the dictionary
  selTrainFeats = [];
  for c = 1:nrClasses
      selTrainFeats = [selTrainFeats vl_colsubset(find(labels==c)', ceil(conf.vocabSubset/nrClasses))] ;      
  end
  
  featureExtractFun = str2func(conf.featureExtractFun);
  
  descrs = {} ;
%   for ii = 1:length(selTrainFeats)
  fprintf('extract features from  %d images for codebook creation\n',length(selTrainFeats));
  tic;
%   for ii = 1:length(selTrainFeats)
  parfor ii = 1:length(selTrainFeats)
%     fprintf('%d %s\n',ii,images{selTrainFeats(ii)});
    descrs{ii} = featureExtractFun(images{selTrainFeats(ii)}, conf);
    descrs{ii}(end+1,:) = labels(selTrainFeats(ii)); 
  end
  toc

  descrs = vl_colsubset(cat(2, descrs{:}), conf.descrsSubset) ;
  descrs_labels = descrs(end,:);
  descrs(end,:) = [];
  descrs = single(descrs) ;
  
  if strcmp(conf.usePCACompression,'yes')
%     conf.usePCACompressionSize = 80;
    if size(descrs,1) < conf.PCACompressionSize
        model.pcaTrans = eye(size(descrs,1));
        model.pcaTransMean = zeros(1,size(descrs,1));
    else
        [COEFF, SCORE] = princomp(descrs');
        model.pcaTrans = COEFF(:,1:conf.PCACompressionSize);
        model.pcaTransMean = mean(descrs,2)';

        descrs = (model.pcaTrans' * (descrs - repmat(model.pcaTransMean,size(descrs,2),1)'));
    end
  end

  % Quantize the descriptors to get the visual words
  if strcmp(conf.codebookClusterAlgorithm,'kmeans')
    fprintf('start clustering \n')
%     if isempty(descrs)
%       vocabulary = single(zeros(size(descrs,1), conf.numWords));
%     else
      vocabulary = vl_kmeans(descrs, conf.numWords, 'verbose', 'algorithm', 'elkan') ;
%     end
  elseif strcmp(conf.codebookClusterAlgorithm,'yael_kmeans')
      vocabulary = yael_kmeans(descrs,conf.numWords,'seed',conf.randSeed);
  elseif strcmp(conf.codebookClusterAlgorithm,'random')
      randIdxs = randsample(size(descrs,2),conf.numWords);
      vocabulary = descrs(:,randIdxs);
  else
      error('unknown clustering algorithm');
  end
  model.vocab = vocabulary;

  if strcmp(conf.codebookCompression,'none')                
      model.map = 1:conf.numWords;
  elseif strcmp(conf.codebookCompression,'aib')                
      [~,bestIdx]=pdist2(vocabulary',descrs', 'euclidean','Smallest',1);
      pcx = zeros(nrClasses, conf.numWords);
      for c = 1:max(labels)
          pcx(c,:) = hist(bestIdx(descrs_labels == c),1:conf.numWords);
      end

      parents=vl_aib(pcx,'ClusterNull');
      [~, map] = vl_aibcut(parents, conf.codebookCompressionSize);
      model.map = map;
  else
      error('unknown option');
  end

if strcmp(conf.quantizer, 'kdtree')
  model.kdtree = vl_kdtreebuild(vocabulary) ;
end


