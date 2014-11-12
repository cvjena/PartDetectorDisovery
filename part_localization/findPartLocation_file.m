function [part_locs,part_visible]=findPartLocation_file(imagepath,bbox, layer,part_channel_assoc,part_id, image_id, locs_file, num_parts)
%     part_locs=-nan(size(part_channel_assoc,1),2);
%     part_visible=zeros(size(part_channel_assoc,1),1);
    load(locs_file,'locs');
    part_locs=locs((image_id-1)*num_parts+1:image_id*num_parts,3:4)+2;
    part_visible=locs((image_id-1)*num_parts+1:image_id*num_parts,5);
end