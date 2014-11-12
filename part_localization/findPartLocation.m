function [part_locs,part_visible]=findPartLocation(imagepath,bbox,imagedir, basedir, layer,part_channel_assoc,part_id)
    part_locs=-nan(size(part_channel_assoc,1),2);
    part_visible=zeros(size(part_channel_assoc,1),1);
    % for the part part_id
    for p=part_id
        % Find the channel which is supposed to be used
        c=part_channel_assoc(p);
        % load required channel gradient maps
        gmap=load(sprintf('%s/%s/gradient_layer%s_channel%i.mat',basedir,imagepath,layer, c-1));
        gmap=gmap.gradient_map;
        if (sum(isnan(gmap(:)))==0 && (length(find(gmap))>10))
            % fit GMM 
            [x,y]=fitGMMToGradient([imagedir '/' imagepath],gmap,bbox,2);
%             % convolve and find max
%             gmap=filter2(fspecial('gaussian',[20 20],4),gmap);
%             [rows,x]=max(gmap,[],2);
%             [~,y]=max(rows,[],1);
%             x=x(y);
%             figure;
%             imshow(gmap/max(gmap(:)));
            % take mean as location
            part_locs(p,:)=[x,y];
            part_visible(p,1)=1;
        else
            part_locs(p,:)=[NaN NaN];
            part_visible(p,1)=0;
        end
    end
end