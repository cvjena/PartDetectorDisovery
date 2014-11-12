% Estimates parts using the nearest neighbor method on HOG features
% The function operates independently from the rest of the code.
% 
% called by ../features/experimentGeneral_extractPartFeatures.m for example
function [parts_test_estimated,parts_train_estimated,confDiffString] = partEstimationDeepLearning(dataset, nrClasses, resDir, conf)

	if nargin < 1
		dataset = 'cub200_2011';
	end
	
	if nargin < 2
		nrClasses = 200;
    end

  if nargin < 3
		resDir = '/tmp/';
	end

	if nargin < 4
		conf = struct([]);
	end

  configDefault.preprocessing_cropToBoundingbox = 1;
  configDefault.preprocessing_standardizeImage = 1;
  configDefault.preprocessing_standardizeImageSize = [128 128];
  configDefault.preprocessing_useMask = 'none';
  configDefault.hog_cellsize = 16;
  configDefault.partEstimation_distanceMeasure = 'euclidean';

  % This parameter should always be 1, otherwise the nearest neighbour
  % search takes the k'th best neighbour, does not make sense, only for debugging.
	configDefault.k=1;

  [config,confDiffString] = parseConfigs(configDefault, conf);

  
    setts = settings();
    train_test_split = load([ setts.dataset_cub200_2011 '/train_test_split.txt' ]);
    train_test_split = train_test_split(:,2);
    parts = load([setts.dataset_cub200_2011 '/parts/est_part_locs.txt' ]);

    parts2 = zeros(length(unique(parts(1,:))), length(unique(parts(2,:))) * 2 );

    for jj = 1:size(parts,1)
        imageId = parts(jj,1);
        partId = parts(jj,2);

        if parts(jj,5) == 1                
            parts2(imageId, 2*partId - 1) = parts(jj,3);
            parts2(imageId, 2*partId ) = parts(jj,4);
        else
            parts2(imageId, 2*partId - 1) = -1;
            parts2(imageId, 2*partId ) = -1;
        end
    end

    parts_train_estimated = parts2(train_test_split == 1, :);
    parts_test_estimated = parts2(train_test_split == 0, :);
end
    
