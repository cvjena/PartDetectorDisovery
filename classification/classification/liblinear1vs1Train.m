function svmmodels = liblinear1vs1Train(hists_train, labels_train, config)
%LIBLINEARTRAIN Summary of this function goes here
%   Detailed explanation goes here
%  addpath([getenv('HOME') '/libs/liblinear-1.92-mem/matlab']);
%   addpath([getenv('HOME') '/libs/liblinear-1.91/matlab']);

  % check if data is ordered according to the labels in a ascending order,
  % if not, reorder (todo: modify liblinear to cope with unordered data?)
  
  [~,sorted_idxs]=sort(labels_train);
  if ~( sum( sorted_idxs' == 1:length(sorted_idxs) ) == length(sorted_idxs))
      % labels are not sorted
      
      hists_train = hists_train(sorted_idxs,:);
      labels_train = labels_train(sorted_idxs);
      
  end

  
  if strcmp( config.svm_Kernel , 'linear')
      hists_train = hists_train';
  elseif strcmp( config.svm_Kernel , 'intersection')
      hists_train = vl_homkermap(hists_train', config.homkermap_n, 'kinters', 'gamma', config.homkermap_gamma) ;
  elseif strcmp( config.svm_Kernel , 'chi-squared')      
      hists_train = vl_homkermap(hists_train', config.homkermap_n, 'kchi2', 'gamma', config.homkermap_gamma) ;
  else
        error('invalid kernel, kernel %s is not impelemented',config.svm_Kernel);
  end
  
  if strcmp(class(hists_train),'single')
      hists_train = double(hists_train);
  end

  nrClasses = max(labels_train);
  
  liblinear_options = sprintf('-o -q -s 3 -c %f -B 1',config.svm_C);
  svmmodels = liblinearTrainMex(labels_train,sparse(hists_train),liblinear_options,'col');
  
%   svmmodels = [];
%   for i = 1:(nrClasses-1)
%       for j = (i+1):nrClasses
%           fprintf('%d %d\n',i,j)
%           currentIdxs = (labels_train == i) | (labels_train == j);
%           
% %           svmmodels(end+1).model = train(int32(labels_train(currentIdxs)), hists_train(:,currentIdxs), liblinear_options, 'col');
%           svmmodels(end+1).model = liblinearTrainMex(double(labels_train(currentIdxs)), sparse(double(hists_train(:,currentIdxs))), liblinear_options, 'col');
%           svmmodels(end).labels = [i j];
%           
% %           current_model
% %           
% %           if isempty(svmmodels)
% %               svmmodels = current_model;
% %           else
% %               svmmodels(end+1) = current_model;
% %           end
%       end
%   end
  
% %   svmmodels = train(double(labels_train), sparse(double(hists_train)), liblinear_options, 'col');
%   
%   [a,b,c]=unique(labels_train);
%   unique_labels = labels_train(sort(b));
%   [~,unique_labels_reverse]=sort(unique_labels);
%   svmmodels.unique_labels_reverse = unique_labels_reverse;
  
    if isfield(config, 'svm_logisticregression') && config.svm_logisticregression == 1 % use cross validation for this?
        error('not implemented')
%         config.svm_Kernel = 'linear'; % features are already mapped
%         [As Bs]=logisticRegression(hists_train', labels_train, ceil(5*randperm(length(labels_train))/length(labels_train)), @liblinearTrainTest, config);
%         
%         svmmodels.As = As;
%         svmmodels.Bs = Bs;
    end
end

