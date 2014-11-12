function [ descrs, frames, imSize ] = extractDecafFeatures( imagename, config )
%EXTRACTDECAFFEATURES Extracts Decaf features given the path to an image
%according to the parameters of the supplied config struct

    [im bbox] = readImage(imagename, config);
    % crop image if not already done
    if ~isfield(config,'preprocessing_cropToBoundingbox') || ~config.preprocessing_cropToBoundingbox
        im = imcrop(im, [bbox.left bbox.top bbox.right-bbox.left bbox.bottom-bbox.top ]);
        bbox.right = bbox.right-bbox.left;
        bbox.top = bbox.bottom-bbox.top;
        bbox.left = 0;
        bbox.top = 0;
    end
    
%     imagename = '/home/ubuntu/decaf/decaf-demo/scene_categories/1/image_0001.jpg';
    setts = settings();
    tmp_file = ['tmp/' char(java.util.UUID.randomUUID()) '.csv'];
    system(['source ' setts.python_env '/bin/activate;python ' setts.libraries_decaf '/imgnet-extract.py --warp --out ' tmp_file ' ' imagename]);
    
    descrs = csvread(tmp_file)';
    frames = repmat([1;1;10;0],1,size(descrs,2)); 
    imSize = size(im);
    
    delete(tmp_file);
end
