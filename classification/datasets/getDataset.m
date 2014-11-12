function [hists_train, labels_train, hists_test, labels_test, hists_realtest] = getDataset(dataset, features, datasetSize)
%GETDATASET Parses datasets into matlab formats
%   Options:
%       dataset = ('nabirds25','stanford_dogs','cu_dogs','dogs','cub200','cub200_2011','caltech101')
%       features = ('imagenames','parts','portmanteau','gps') (if available)
%       datasetSize = dataset specific

  setts = settings();

  hists_train = [];
  labels_train = [];
  hists_test = [];
  labels_test = [];

  if strcmp(dataset, 'cub200_2011')
        dataBaseDir = [setts.dataset_cub200_2011 '/'];

        train_test_split = load([ dataBaseDir '/train_test_split.txt' ]);
        train_test_split = train_test_split(:,2);

        labels = load([ dataBaseDir '/image_class_labels.txt' ]);
        labels = labels(:,2);
        
        labels_train = labels(train_test_split == 1);
        labels_test = labels(train_test_split == 0);

        if strcmp(features, 'imagenames')
            fid = fopen([ dataBaseDir '/images.txt' ]);
            images = textscan(fid, '%s %s');
            fclose(fid);
            images = images{2};
            
            images = strcat([setts.dataset_cub200_2011 '/images/'],images);

            
            hists_train = images(train_test_split == 1);
            hists_test = images(train_test_split == 0);
            
            

        elseif  strcmp(features, 'parts')
            parts = load([ dataBaseDir 'parts/part_locs.txt' ]);
            
            parts2 = zeros(length(unique(parts(1,:))), length(unique(parts(2,:))) * 2 );
            
            for jj = 1:size(parts,1)
                imageId = parts(jj,1);
                partId = parts(jj,2);
                
                if parts(jj,5) == 1                
                    parts2(imageId, 2*partId - 1) = parts(jj,3);
                    parts2(imageId, 2*partId ) = parts(jj,4);
                else
                    parts2(imageId, 2*partId - 1) = -1;
                    parts2(imageId, 2*partId ) = -1;
                end
            end
            
            hists_train = parts2(train_test_split == 1, :);
            hists_test = parts2(train_test_split == 0, :);
            
        elseif  strcmp(features, 'bboxes')
            cachedir = [setts.cachedir '/' 'CUB_200_2011' '/']
            if ~exist(cachedir,'dir')
                mkdir(cachedir);
            end
            if ~exist([cachedir '/bboxes_cache.mat'],'file')
                [images_train, ~, images_test, ~] = getDataset('cub200_2011','imagenames',200);

                for ii = 1:length(images_test)
                    bbox = readBbox(images_test{ii});
                    hists_test{ii} = bbox;
                end

                for ii = 1:length(images_train)
                    bbox = readBbox(images_train{ii});
                    hists_train{ii} = bbox;
                end
                hists_train = hists_train';
                hists_test = hists_test';

                save([cachedir '/bboxes_cache.mat'],'hists_train','hists_test');
            else
                load([cachedir '/bboxes_cache.mat']);
            end
            
        end
        
      if datasetSize == 14
        classesnr = [36 151 152 153 154 155 156 157 187 188 189 190 191 192];
        train_idxs = ismember(labels_train, classesnr);
        test_idxs = ismember(labels_test, classesnr);

        labels_train = labels_train(train_idxs);
        labels_test = labels_test(test_idxs);
        hists_train = hists_train(train_idxs, :);
        hists_test = hists_test(test_idxs, :);
        
        %remap labels to make them continous
        labelmap(classesnr) = 1:length(classesnr);
        labels_train = labelmap(labels_train)';
        labels_test = labelmap(labels_test)';
      elseif datasetSize == 3
        classesnr = [1 2 3];
        train_idxs = ismember(labels_train, classesnr);
        test_idxs = ismember(labels_test, classesnr);

        labels_train = labels_train(train_idxs);
        labels_test = labels_test(test_idxs);
        hists_train = hists_train(train_idxs, :);
        hists_test = hists_test(test_idxs, :);
        
        %remap labels to make them continous
        labelmap(classesnr) = 1:length(classesnr);
        labels_train = labelmap(labels_train)';
        labels_test = labelmap(labels_test)';
      elseif datasetSize == 200

      else
        assert(false, 'size not implemented')            
      end
  else
    assert(false, 'unknown dataset')  
  end

  

end
