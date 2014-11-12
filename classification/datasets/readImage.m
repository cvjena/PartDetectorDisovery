function [im bbox] = readImage(imageName, config)
%READIMAGE Reads image and bbox from file, applies preprocessing
%   detailed description.

    im = imread(imageName);
    bbox = readBbox(imageName, config);

    if( size(im,3) == 1)
        imrgb(:,:,1) = im;
        imrgb(:,:,2) = im;
        imrgb(:,:,3) = im;
        im = imrgb;
    end
    
    if isfield(config,'preprocessFunction')
        [im,bbox] = config.preprocessFunction(im, bbox, imageName, config);
    end

    if isfield(config,'preprocessing_cropToBoundingbox') && config.preprocessing_cropToBoundingbox
        x_min = max(1, bbox.left - config.preprocessing_boundingboxEnlargement);
        x_max = min(size(im,2), bbox.right + config.preprocessing_boundingboxEnlargement);
        y_min = max(1, bbox.top - config.preprocessing_boundingboxEnlargement);
        y_max = min(size(im,1), bbox.bottom + config.preprocessing_boundingboxEnlargement);
        im = imcrop(im, [ x_min y_min x_max-x_min y_max-y_min]);
        bbox.right = bbox.right-x_min;
        bbox.top = bbox.bottom-y_min;
        bbox.left = x_min;
        bbox.top = y_min;
    end

    if isfield(config,'preprocessing_flipImage') && config.preprocessing_flipImage == 1
        im = flipdim(im,2);
    end
    
    if isfield(config,'preprocessing_standardizeImage') && config.preprocessing_standardizeImage
        if length(config.preprocessing_standardizeImageSize) == 1
            if max(size(im)) > config.preprocessing_standardizeImageSize 
                scale_factor = config.preprocessing_standardizeImageSize / max(size(im));
                bbox.left = bbox.left * scale_factor;
                bbox.right = bbox.right * scale_factor;
                bbox.bottom = bbox.bottom * scale_factor;
                bbox.top = bbox.top * scale_factor;

                im = imresize(im, scale_factor ); 
            end
        elseif length(config.preprocessing_standardizeImageSize) == 2
            imdims = size(im);
            scale_factor = config.preprocessing_standardizeImageSize ./ imdims(1:2);

            bbox.left = bbox.left * scale_factor(1);
            bbox.right = bbox.right * scale_factor(1);
            bbox.bottom = bbox.bottom * scale_factor(2);
            bbox.top = bbox.top * scale_factor(2);

            im = imresize(im, config.preprocessing_standardizeImageSize );      
            % assert(1==0) % is anything using this? if yes, make sure bounding box matches
        end
    end
    
end

