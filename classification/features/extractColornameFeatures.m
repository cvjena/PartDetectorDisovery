% Extracts features given the path to an image
function [ descrs, frames, imSize ] = extractColornameFeatures( imagename, config )

    [im bbox] = readImage(imagename, config);
    if ~isa(im, 'double')
      im = double(im);
    end

    imSize = size(im);

    start = round(config.colornameFeature_width/2) + 1;
    step = config.colornameFeature_step;

    if size(im,1) <= 2*start || size(im,2) <= 2*start 
        descrs = double(zeros(11,0)); frames = double(zeros(4,0)); imSize = size(im);
        return
    end

    w2c = [];
    load('w2c.mat');
    
    cim = [];
    for i = 1:11
        cim(:,:,i)=im2c(im,w2c,i);
    end

    intim = cumsum(cumsum(double(cim),1),2);
    intim = [ zeros(1,size(intim,2)+1,11); [zeros(size(intim,1),1,11) intim]]; % pad with one row/column of zeros

    
    width = config.colornameFeature_width;
%     height = config.simplefeature_height;
    resIm = computeMeanBox(intim, width, width, round(width/2), round(width/2));
    
    
    
    [a,b]=ind2sub([size(resIm,1) size(resIm,2)],1:(size(resIm,1)*size(resIm,2)) );
    frames = [b' a' zeros(length(a),2) ]';
    descrs = reshape(resIm, length(a),11)';
    
    xIdxs = mod((frames(1,:)-start)/step,1)==0 & frames(1,:) >= start & frames(1,:) <= (size(resIm,2)-start+1);
    yIdxs = mod((frames(2,:)-start)/step,1)==0 & frames(2,:) >= start & frames(2,:) <= (size(resIm,1)-start+1);
    
    frames = frames(:,xIdxs & yIdxs);
    descrs = descrs(:,xIdxs & yIdxs);
    
    if ~strcmp(config.preprocessing_useMask,'none')
        mask = readMask(imagename, config);
        ind = sub2ind(size(im),frames(2,:), frames(1,:));
        maskIdxs = mask(ind);
        descrs = descrs(:,maskIdxs);
        frames = frames(:,maskIdxs);
    end

    if isfield(config,'preprocessing_excludeMask') && config.preprocessing_excludeMask
        mask = readMask(imagename, config);
        ind = sub2ind(size(im),frames(2,:), frames(1,:));
        maskIdxs = mask(ind);
        descrs = descrs(:,~maskIdxs);
        frames = frames(:,~maskIdxs);
    end

    if isfield(config,'excludeGTBoundingBoxes') && config.excludeGTBoundingBoxes
        idxs_inside = frames(1,:) >= bbox.left & frames(1,:) <= bbox.right & frames(2,:) >= bbox.top & frames(2,:) <= bbox.bottom;
        descrs = descrs(:,~idxs_inside);
        frames = frames(:,~idxs_inside);
    end

    imSize = size(im);

end


function [paddedResIm, mask] = computeMeanBox(intim, sizeX, sizeY,displacementX, displacementY)
    imSize = size(intim) - [1 1 0];

%     im1 = imcrop(intim, [1 1 imSize(2)-sizeX imSize(1)-sizeY]);
%     im2 = imcrop(intim, [sizeX+1 sizeY+1 imSize(2)-sizeX imSize(1)-sizeY]);
%     im3 = imcrop(intim, [1 sizeY+1 imSize(2)-sizeX imSize(1)-sizeY]);
%     im4 = imcrop(intim, [sizeX+1  1 imSize(2)-sizeX imSize(1)-sizeY]);

    im1 = intim(1:(imSize(1)-sizeY+1), 1:(imSize(2)-sizeX+1),:);
    im2 = intim((sizeY+1):(imSize(1)+1), (sizeX+1):(imSize(2)+1),:);
    im3 = intim((sizeY+1):(imSize(1)+1), 1:(imSize(2)-sizeX+1),:);
    im4 = intim(1:(imSize(1)-sizeY+1), (sizeX+1):(imSize(2)+1),:);
    
    resIm = im2 + im1 - im3 - im4;
    
    paddedResIm = zeros(imSize);
    mask= false(imSize);
    
    yIdxsDest = displacementY + (1:size(resIm,1));
    xIdxsDest = displacementX + (1:size(resIm,2));
    yIdxsSrc = 1:size(resIm,1);
    xIdxsSrc = 1:size(resIm,2);
    
    t = yIdxsDest < 1 | yIdxsDest > imSize(1) | yIdxsSrc < 1 | yIdxsSrc > imSize(1);
    yIdxsDest(t) = [];
    yIdxsSrc(t) = [];

    t = xIdxsDest < 1 | xIdxsDest > imSize(2) | xIdxsSrc < 1 | xIdxsSrc > imSize(2);
    xIdxsDest(t) = [];
    xIdxsSrc(t) = [];
    
    if ~( isempty(xIdxsDest) | isempty(yIdxsDest))
        paddedResIm(yIdxsDest, xIdxsDest,:) = resIm(yIdxsSrc, xIdxsSrc,:);
        mask(yIdxsDest, xIdxsDest,:) = 1;
    end

end
