% Estimates parts using the nearest neighbor method on HOG features
% The function operates independently from the rest of the code.
% 
% called by ../features/experimentGeneral_extractPartFeatures.m for example
function [parts_test_estimated,parts_train_estimated,confDiffString] = partEstimationNN(dataset, nrClasses, resDir, conf)

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
  configDefault.preprocessing_boundingboxEnlargement = 0;
  configDefault.preprocessing_useMask = 'none';
  configDefault.hog_cellsize = 16;
  configDefault.partEstimation_distanceMeasure = 'euclidean';

  % This parameter should always be 1, otherwise the nearest neighbour
  % search takes the k'th best neighbour, does not make sense, only for debugging.
	configDefault.k=1;

  [config,confDiffString] = parseConfigs(configDefault, conf);

  % mfilename specifies the filename of the function
	featureCacheFile = [resDir 'cache_' mfilename confDiffString '.mat']; 
	if exist(featureCacheFile,'file')
        fprintf('Loading HOG features from %s\n', featureCacheFile);
        load(featureCacheFile,'parts_test_estimated','parts_train_estimated');
    else
        fprintf('Cache file %s does not exist. Start computing HOG features.\n',featureCacheFile);
        tic;
	
        [images_train, labels_train, images_test, labels_test] = getDataset(dataset,'imagenames',nrClasses);
        [parts_train, ~, parts_test, ~] = getDataset(dataset,'parts',nrClasses);
        [bbox_train, ~, bbox_test, ~] = getDataset(dataset,'bboxes',nrClasses);
	
        hog_train = [];
        hog_train_flipped = [];

        % loop through all training images
        parfor ii = 1:length(images_train)
    %   for ii = 1:length(images_train)
            if mod(ii,10) == 0
                fprintf('HOG Calculation for all train images %d/%d\n',ii,length(images_train));
            end
            image_name = images_train{ii};
    
            im = readImage(image_name, config);
            mask = readMask(image_name, config);
            mask(:,:,2) = mask(:,:,1);
            mask(:,:,3) = mask(:,:,1);
            % simple HOG masking
            im(mask==0)=0;
            
            hog = vl_hog(im2single(im),config.hog_cellsize);
            hog_train(ii,:) = hog(:);
    
            im_flipped = flipdim(im,2);
            hog = vl_hog(im2single(im_flipped),config.hog_cellsize);
            hog_train_flipped(ii,:) = hog(:);
        end
	    
	    
        hog_test = [];
	      
        % loop through all test images
        parfor ii = 1:length(images_test)
    %     for ii = 1:length(images_test)
            if mod(ii,100) == 0
                fprintf('HOG Calculation for all test images %d/%d\n',ii,length(images_test));
            end
            image_name = images_test{ii};
    
            im = readImage(image_name, config);
            mask = readMask(image_name, config);
            mask(:,:,2) = mask(:,:,1);
            mask(:,:,3) = mask(:,:,1);
            % simple HOG masking
            im(mask==0)=0;

            hog = vl_hog(im2single(im),config.hog_cellsize);
            hog_test(ii,:) = hog(:);
        end
	    
        hog_train_both = [hog_train; hog_train_flipped];
        
        % perform nearest neighbour search based on HOG
        [a]=pdist2(hog_train_both,hog_test,config.partEstimation_distanceMeasure);
        [d,e]=sort(a);    
        % get the k'th neighbour
        c = e(config.k,:);
% 	    [b,c]=min(a);
	    
	
        parts_test_estimated = [];
	
        % loop through all test images and set the parts
        for ii = 1:length(images_test)
	%        imtest = readImage(images_test{ii},config);
	%        imnn = readImage(images_train{mod( c(ii) - 1, length(images_train))+1},config);

          % get the training image that matches with ignoring flipped images (therefore the mod)
	        bestIdx = mod( c(ii) - 1, length(images_train))+1;

          % check whether we matched with a flipped training image
	        flip = c(ii) > length(images_train);

          % get the parts from the nearest image
	        current_parts = parts_train(bestIdx,:);

          % TODO
          % This is the code part where we need a better part transfer procedure
          %


          % normalize the parts
          % we have 15 parts, therefore the repmat
          % the parts are shifted to the origin defined by the bounding box
          % and normalized to 0, 1 coordinates
	        current_parts_norm = (current_parts - repmat([bbox_train{bestIdx}.left bbox_train{bestIdx}.top],1,15)) ./ repmat([bbox_train{bestIdx}.right-bbox_train{bestIdx}.left bbox_train{bestIdx}.bottom-bbox_train{bestIdx}.top],1,15);        
          
          % handle unknown parts
	        current_parts_norm(current_parts == -1) = -1;

          % take care of part flipping
	        if flip
              % left wing is now a right wing, and so on
	            current_parts_norm(1:2:end) = 1 - current_parts_norm(1:2:end);
	            current_parts_norm(current_parts == -1) = -1;
	            tmp = current_parts_norm(13:18);
	            current_parts_norm(13:18) = current_parts_norm(21:26);
	            current_parts_norm(21:26) = tmp;
	        end
	        
	        % transfer the parts to the current image
          % first: scale the parts with the current bounding box size
          % second: translate it using the current bounding box origin
	        current_parts_norm_trans = (current_parts_norm .* repmat([bbox_test{ii}.right-bbox_test{ii}.left bbox_test{ii}.bottom-bbox_test{ii}.top],1,15)) + repmat([bbox_test{ii}.left bbox_test{ii}.top],1,15);        

          % again handle unknown parts
	        current_parts_norm_trans(current_parts_norm==-1) = -1;
	        current_parts_norm_trans = round(current_parts_norm_trans);
	       
          % add the parts
	        parts_test_estimated(ii,:) = current_parts_norm_trans;
	        
          % perform some visualization (you also have to uncomment the above statements)
	%        if c(ii) > length(images_train)
	%            imnn = flipdim(imnn,2);
	%        end
	%        c(ii)
	%        
	%        subplot(1,2,1);
	%        imshow(imtest);
	%        subplot(1,2,2);
	%        imshow(imnn);
	%        waitforbuttonpress
	    end
	    
	   
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      % now "estimate" parts for the training image
      % TODO: why???

	    [a]=pdist2(hog_train_both,hog_train);
	    % [b,c]=min(a);
	    [d,e]=sort(a);   
      % get the second and third best, because the best
      % neighbour is always the example itself
      % completely ignores the setting of k
      % but the following two statements are overwritten anyway!
	    c = e(2,:); % useless TODO
	    c2 = e(3,:); % useless TODO
	    
      parts_train_estimated = [];
	
	    for ii = 1:length(images_train)
	    	% get all matches 
        % best match should be the image itself for all non-flipped images
        tmpcc = e(:,ii);
	    	% delete self-matches from the list (either flipped or non-flipped)
        tmpcc(tmpcc == ii | tmpcc == ii+length(images_train)) = [];
        % now we have the k'th neighbour for the ii'th training image, which is not ii itself
	    	c(ii) = tmpcc(config.k);
	    	
        % for visualization only
        % imtest = readImage(images_test{ii},config);
        % imnn = readImage(images_train{mod( c(ii) - 1, length(images_train))+1},config);

        % get the training index of the match
        bestIdx = mod( c(ii) - 1, length(images_train))+1;
        flip = c(ii) > length(images_train);
       
        % if this is still the image itself, we have a problem
        if ii == bestIdx
            bestIdx = mod( c2(ii) - 1, length(images_train))+1;
            flip = c2(ii) > length(images_train);
            
            assert(ii ~= bestIdx);
        end
        
        % transfer the parts of the match (some code as before)
        current_parts = parts_train(bestIdx,:);
        current_parts_norm = (current_parts - repmat([bbox_train{bestIdx}.left bbox_train{bestIdx}.top],1,15)) ./ repmat([bbox_train{bestIdx}.right-bbox_train{bestIdx}.left bbox_train{bestIdx}.bottom-bbox_train{bestIdx}.top],1,15);        
        current_parts_norm(current_parts == -1) = -1;
        if flip
            current_parts_norm(1:2:end) = 1 - current_parts_norm(1:2:end);
            current_parts_norm(current_parts == -1) = -1;
            tmp = current_parts_norm(13:18);
            current_parts_norm(13:18) = current_parts_norm(21:26);
            current_parts_norm(21:26) = tmp;
        end
	        
	        
        current_parts_norm_trans = (current_parts_norm .* repmat([bbox_train{ii}.right-bbox_train{ii}.left bbox_train{ii}.bottom-bbox_train{ii}.top],1,15)) + repmat([bbox_train{ii}.left bbox_train{ii}.top],1,15);        
        current_parts_norm_trans(current_parts_norm==-1) = -1;
        current_parts_norm_trans = round(current_parts_norm_trans);
        
        parts_train_estimated(ii,:) = current_parts_norm_trans;
	       
        % visualization
      	%        if c(ii) > length(images_train)
      	%            imnn = flipdim(imnn,2);
        %        end
        %        c(ii)
        %        
        %        subplot(1,2,1);
        %        imshow(imtest);
        %        subplot(1,2,2);
        %        imshow(imnn);
        %        waitforbuttonpress
	    end
	   
      fprintf('Saving the estimated parts in %s\n', featureCacheFile);
	    save(featureCacheFile,'parts_test_estimated','parts_train_estimated');
    end
end
    
