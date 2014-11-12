function mask = readMask(imageName, config)
%READMASK Reads mask from file, applies preprocessing
%   Options:
%       config.preprocessing_useMask = ('grabcut', 'bbox', 'gt', 'none')
%           'grabcut' - uses grabcut to compute mask from gt-bbox
%           'bbox' - uses gt-bounding-box
%           'gt' - uses ground truth segmentation, if available
%           'none' - uses the whole image
%
%       config.preprocessing_cropToBoundingbox = (True,False)
%           - crops mask to bbox
%
%       config.preprocessFunction = @function(im,bbox,path,config)
%           - applies this function to every mask
%
%       config.preprocessing_standardizeImage = (0,1,2)
%           0 - uses unaltered image
%           1 - scales largest dim to config.preprocessing_standardizeImageSize
%           2 - scales image to config.preprocessing_standardizeImageSize =
%           [ysize, xsize]
%
    setts = settings();

    bbox = readBbox(imageName, config);

	if isfield(config,'preprocessing_useMask') && strcmp(config.preprocessing_useMask,'grabcut')
	    if ~isempty(strfind(imageName,'CUB_200_2011'))
		    maskName = strrep(imageName, setts.dataset_cub200_2011, [setts.cachedir '/CUB_200_2011/grabcut/']);
		    maskName = strrep(maskName, 'jpg', 'png');
		else
			assert(false,'no grabcut masks implemented for this dataset');
		end

		if ~exist(maskName,'file')
      fprintf('re-creating mask: %s %d\n', maskName, exist(maskName, 'file'));
			bbox = readBbox(imageName,{});
			im = readImage(imageName,{});
      try
			  mask = grabCutMex(im,[bbox.left bbox.top bbox.right-bbox.left bbox.bottom-bbox.top]);
      catch exc
        fprintf('Error in grabCut method, using the bounding box as a fallback solution\n');
        fprintf('Image filename:\n');
        disp(imageName);
        fprintf('Bounding box:\n');
        disp(bbox);
        fprintf('Image size:\n');
        disp(size(im))
        getReport(exc);
        mask = zeros(size(im,1),size(im,2));
        mask((bbox.top+1):bbox.bottom, (bbox.left+1):bbox.right) = 255;
      end
			slashpos = strfind(maskName,'/');
			outputdir = maskName(1:(slashpos(end)-1));
			if ~exist(outputdir,'dir')
			    mkdir(outputdir);
			end
			imwrite(255*mod(mask,2),maskName,'PNG');
		end

	    mask = imread( maskName );		

		maskBbox = imcrop(mask, [bbox.left bbox.top bbox.right-bbox.left bbox.bottom-bbox.top ]);
		if(sum(maskBbox(:) > 128) <= 0.1*numel(maskBbox) ) % empty mask, use bounding box instead
		    if isfield(config,'preprocessing_cropToBoundingbox') && config.preprocessing_cropToBoundingbox
		        mask(:,:) = 255;
		    else
		        mask((bbox.top+1):bbox.bottom, (bbox.left+1):bbox.right) = 255;
		    end
		end

	elseif isfield(config,'preprocessing_useMask') && strcmp(config.preprocessing_useMask,'gt')
		if ~isempty(strfind(imageName,'CUB_200_2010')) 
		    maskName = strrep(imageName, '/CUB_200_2010/images/', '/CUB_200_2010/annotations-mat/');
		    maskName = strrep(maskName, 'jpg', 'mat');
		else
			assert(false,'no gt masks implemented for this dataset');
		end
		anno = load( maskName );
        mask = anno.seg * 255;
	elseif isfield(config,'preprocessing_useMask') && strcmp(config.preprocessing_useMask,'bbox')
        im = imread(imageName);
        mask = zeros(size(im,1),size(im,2));
        mask((bbox.top+1):bbox.bottom, (bbox.left+1):bbox.right) = 255;
    else
        im = imread(imageName);
        mask = zeros(size(im,1),size(im,2));
		mask(:) = 255;
	end


    if isfield(config,'preprocessFunction')
        [mask, bbox] = config.preprocessFunction(mask, bbox, imageName, config);
    end
 

    if isfield(config,'preprocessing_cropToBoundingbox') && config.preprocessing_cropToBoundingbox
        mask = imcrop(mask, [bbox.left bbox.top bbox.right-bbox.left bbox.bottom-bbox.top ]);
        bbox.right = bbox.right-bbox.left;
        bbox.top = bbox.bottom-bbox.top;
        bbox.left = 0;
        bbox.top = 0;
    end
    
    if isfield(config,'preprocessing_flipImage') && config.preprocessing_flipImage == 1
        mask = flipdim(mask,2);
    end

    if isfield(config,'preprocessing_standardizeImage') && config.preprocessing_standardizeImage
        if length(config.preprocessing_standardizeImageSize) == 1
            if max(size(mask)) > config.preprocessing_standardizeImageSize 
                scale_factor = config.preprocessing_standardizeImageSize / max(size(mask));
                mask = imresize(mask, scale_factor ); 
            end
        elseif length(config.preprocessing_standardizeImageSize) == 2
            imdims = size(mask);
            mask = imresize(mask, config.preprocessing_standardizeImageSize );      
        end
    end
    
    mask = (mask > 128);
    

end
