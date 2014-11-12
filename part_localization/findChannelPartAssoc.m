function [ part_cannnel_assoc ] = findChannelPartAssoc( imagelist, parts, dataset_dir, basedir, layer )  
%     dists= table('VariableNames',{'image' 'part' 'channel' 'dist'});
%     data=load('dists-1-2202-0768b1db-c32f-4347-b0fe-6d99547748f3.mat');
%     dists=data.dists;

    bboxes=dlmread([dataset_dir '../bounding_boxes.txt']);
    dists=nan(size(imagelist,1),15,256);
    start=1;
    stop=size(imagelist,1);
    nextsave=start+20000;
    image_ids=unique(parts.image);
    start_time=tic;
    est_channel_locs = nan(size(imagelist,1),256,2);
    parfor j=start:stop
        i=image_ids(j);
        disp(j);
        for c=1:256
            %find mean value
            %calc distance
            %store in (part,channel,image)=distance matrix
            %read gradient map
            gmap=load(sprintf('%s%s/gradient_layer%s_channel%i.mat',basedir, imagelist{j},layer, c-1));
            gmap=gmap.gradient_map;
            if (sum(isnan(gmap(:)) )>0 || sum(gmap(:)~=0)<1)
                continue;
            end
%             tic
            [est_x,est_y]=fitGMMToGradient([dataset_dir imagelist{j}],gmap,bboxes(i,2:end),2);
            est_channel_locs(j,c,:)=[est_x,est_y];
%             toc
            for p=1:15
%                 fprintf('Channel %i part %i\n',c,p);
                line=(j-1)*15+p;
                assert(parts.image(line)==i&&parts.part(line)==p);
                x=parts.x(line);
                y=parts.y(line);
                if (parts.visible(line)&&~isnan(est_x))
                    % location of the strongest component is stored in model.mu(:,1)
                    dists(j,p,c)=sqrt((double(est_y)-double(y))^2+(double(est_x)-double(x))^2)/(sqrt(bboxes(i,4)^2+bboxes(i,5)^2));
                    
%                     disp(p)
%                     disp(dists(i,p,c+1))
%                     plot(x,y,'o','MarkerSize',20,'LineWidth',3);
%                     ginput(1);
                end
            end
%             close all;
        end
%         if (j>nextsave)
%             save(sprintf('dists-%i-%i-%s.mat',start,j,char(java.util.UUID.randomUUID)),'dists');
%             nextsave=nextsave+1000;
%         end
        toc(start_time);
    end
%     save(sprintf('dists-%i-%i-%s.mat',start,stop,char(java.util.UUID.randomUUID)),'dists');
    save(sprintf('est_channel_locs-%i-%i-%s.mat',start,stop,char(java.util.UUID.randomUUID)),'est_channel_locs');
    part_channel_dists=nanmedian(dists,1);
    [~,part_cannnel_assoc]=min(part_channel_dists,[],3);
    part_cannnel_assoc=part_cannnel_assoc';
end

