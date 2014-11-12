function [ descrs, frames, imSize ] = extractSiftFeatures( imagename, config )
%EXTRACTSIFTFEATURES Extracts SIFT features given the path to an image
%according to the parameters of the supplied config struct

    [im bbox] = readImage(imagename, config);
    
    if size(im,1) <= 10 || size(im,2) <= 10 
        siftColorOptionIdx = find(strcmp(config.phowOpts,'Color')) + 1;
        siftColorOption = config.phowOpts{siftColorOptionIdx};
        
        if strcmp(siftColorOption,'opponent')        
            descrs = uint8(zeros(384,0)); frames = double(zeros(4,0)); imSize = size(im);
        elseif strcmp(siftColorOption,'gray')
            descrs = uint8(zeros(128,0)); frames = double(zeros(4,0)); imSize = size(im);
        else
            error('not implemented')
        end
        return
    end
    
    [frames, descrs] = vl_phow(im2single(im), config.phowOpts{:});
    
    if isfield(config,'useRootSIFT') && strcmp(config.useRootSIFT,'yes')
%         descrs = sqrt(descrs);
        descrs = uint8(sqrt(single(descrs))*sqrt(255));
    end
    
    if ~strcmp(config.preprocessing_useMask,'none')
        mask = readMask(imagename, config);
        
        if isfield(config,'preprocessing_excludeMask') && config.preprocessing_excludeMask
            mask = ~mask;
        end
        
        % check if coordinates are integer. if not, round them
        if sum(sum(abs(frames(1:2,:) - round(frames(1:2,:))))) > 0 
            ind = sub2ind(size(im),round(frames(2,:)), round(frames(1,:)));
        else
            ind = sub2ind(size(im),frames(2,:), frames(1,:));
        end
        
        maskIdxs = mask(ind);
        descrs = descrs(:,maskIdxs);
        frames = frames(:,maskIdxs);
    end

    imSize = size(im);

end

