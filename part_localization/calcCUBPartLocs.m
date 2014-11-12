function [  ] = calcCUBPartLocs(  )
    output_dir='results';
    mkdir(output_dir);
    
    % path to gradient maps of this dataset 
    basedir='/home/simon/tmp/cub200-maps/';
    % path to cub200 images
    imagedir='/home/simon/Datasets/CUB_200_2011/images/';
    % imagelist with paths relative to imagedir
    fid=fopen('/home/simon/Datasets/CUB_200_2011/imagelist.txt','r');
    imagelist=textscan(fid,'%s');
    imagelist=imagelist{1};
    fclose(fid);
    % load train test split
    train_test=logical(load('/home/freytag/data/finegrained/cub200_2011/cropped_256x256_flipped/train_test_split.txt'));
    %read part locations
    parts = readtable('/home/simon/Datasets/CUB_200_2011/parts/part_locs.txt','Delimiter',' ','ReadVariableNames',false);
%     parts = readtable('/home/simon/Datasets/CUB_200_2011/parts/est_part_locs.txt','Delimiter',' ','ReadVariableNames',false);
    parts.Properties.VariableNames{'Var1'} = 'image';
    parts.Properties.VariableNames{'Var2'} = 'part';
    parts.Properties.VariableNames{'Var3'} = 'x';
    parts.Properties.VariableNames{'Var4'} = 'y';
    parts.Properties.VariableNames{'Var5'} = 'visible';
    % read bbox
    bboxes=dlmread([imagedir '../bounding_boxes.txt']);
    % layer
    layer = 'pool5';
    train_images=imagelist(logical(train_test(:,2)));
    % all part locs
    est_part_locs=zeros(5*size(imagelist,1),5);
    
    %% Training
    if exist('part_channel_assoc.mat', 'file')
        % Skip channel discovery if we already have part channel association
        data=load('part_channel_assoc.mat');
        part_channel_assoc=data.part_channel_assoc;
    else
        % find relevation channels
        part_selector=repmat(logical(train_test(:,2))',15,1);
        [ part_channel_assoc ] = findChannelPartAssoc(train_images,parts(part_selector(:),:),imagedir,basedir,layer);
        save(['part_channel_assoc-' char(java.util.UUID.randomUUID) '.mat'],'part_channel_assoc');
    end
    
    %% Prediction
    tic
    last_image=nan;
    % Now we know which channels belongs to which part
    % Use that to predict the part locatoin for (all/testing) images
    if (true)
        % Columns of locs correspond to image_id, part_id, x, y, is_visible
        locs = zeros(size(imagelist,1)*numel(part_channel_assoc),5);
        for j=1:size(locs,1)%(randi(2000)*15+16):(11788*15)%(1861*15+1):(11788*15)
            i=int32(floor((j-1)/15))+1;
            p=mod(j-1,size(part_channel_assoc,1))+1;
            % skip training images
%             if (train_test(i,2)==1)
%                 continue;
%             end
            fprintf('Image %i Part %i after %f\n',i,p, toc);
            [part_locs,part_visible]=findPartLocation([imagelist{i}],bboxes(i,2:end), imagedir, basedir,layer,part_channel_assoc,p);
            last_image=i;
            if (true) %any (p==[1 2 3 9 14]))
                % store the location
                locs(j,:)=[i p part_locs(p,1) part_locs(p,2) part_visible(p)];
            end
        end
        dlmwrite('est_part_locs.txt',locs,' ');
        save('est_part_locs.mat','locs');
    end
    
    %% Evaluation
    % matrix for storing distance between real and estimated part locations
    dists=nan(size(imagelist,1)*size(part_channel_assoc,1),1);
    visibility=nan(size(imagelist,1)*size(part_channel_assoc,1),1);
    tic
    % for each image
    % very good: ch 7 for 1030+
    last_image=nan;
    for j=1:numel(dists)%(randi(2000)*15+16):(11788*15)%(1861*15+1):(11788*15)
        % Determine image and part id
        i=int32(floor((j-1)/15))+1;
        p=mod(j-1,size(part_channel_assoc,1))+1;
        % skip training images
        if (train_test(i,2)==1)
            continue;
        end
        if (true) %last_image~=i)
            if (mod(i,1000)==1)
                fprintf('Image %i Part %i after %f\n',i,p, toc);
            end
            [part_locs,part_visible]=findPartLocation_file( imagelist{i},bboxes(i,2:end),layer,part_channel_assoc,p,i,'est_part_locs.mat',numel(part_channel_assoc));
            last_image=i;
        end
        % For visualization
%         if (p==1)
%             tic
%             close all;
%             figure;
%             img=imread(sprintf('%s/%s',imagedir, imagelist{i}));
% %             img=imresize(img,[227 227],'bicubic');
%             imshow(img);
%             hold all;
%         end
        if (true) %any (p==[1 2 3 9 14]))
            % find the location
            est_part_locs(j,:)=[i,p,part_locs(p,1),part_locs(p,2),part_visible(p)];
            visibility(j)=part_visible(p)==parts.visible(j);
            if (part_visible(p)&&parts.visible(j))
                dists(j)=sqrt((part_locs(p,1)-parts.x(j))^2+(part_locs(p,2)-parts.y(j))^2)/(sqrt(bboxes(i,4)^2+bboxes(i,5)^2));
            end
            % For visualization
%             col=hsv(15);
%                 part_locs(p,1),part_locs(p,2)
%             plot(part_locs(p,1),part_locs(p,2),'x','MarkerSize',20,'LineWidth',3,'color',col(p,:));
%                 parts.x(index),parts.y(index)
%             plot(parts.x(j),parts.y(j),'o','MarkerSize',20,'LineWidth',3,'color',col(p,:));
%             [x2,y2]=ginput(1)
%             sqrt((x1-x2)^2+(y1-y2)^2)
%             fprintf('Bounding box size: %f',(sqrt(bboxes(i,4)^2+bboxes(i,5)^2)));
%             close all;
        end
        % For visualization
%         if (p==15)
% %             toc
%             fprintf('%f\n',dists(j));
%             ginput(1);
%         end
    end
%     save(sprintf('%s/est_locs-final-%s.mat',output_dir,char(java.util.UUID.randomUUID)),'est_part_locs');
%     save([output_dir '/part_results-' char(java.util.UUID.randomUUID) '.mat'],'visibility','dists');
    disp('Total average distance from real location:');
    disp(nanmean(dists(:)));
    disp('Average distance from real location per part:');
    dists=reshape(dists,15,11788);
    disp(nanmean(dists,2));
end

