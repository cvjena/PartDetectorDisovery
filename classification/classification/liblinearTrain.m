% Training a model with liblinear and explicit feature transformation
%
%   addpath([getenv('HOME') '/libs/liblinear-1.92-mem/matlab']);
%   addpath([getenv('HOME') '/libs/liblinear-1.91/matlab']);
function svmmodels = liblinearTrain(hists_train, labels_train, config)

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
  
%   if strcmp(class(hists_train),'double')
%       hists_train = single(hists_train);
%   end
  
  liblinear_options = sprintf('-s 7 -c 0.005 -p 0.1');%',config.svm_C);
%   liblinear_options = '-t 0';
  fprintf('liblinear training ...\n');
  svmmodels = train(double(labels_train), sparse(double(hists_train)'), liblinear_options);
% svmmodels = liblinearTrainMex(double(labels_train), sparse(double(hists_train)), liblinear_options, 'col');
% svmmodels = liblinearTrainMex_dense_float(double(labels_train), ((hists_train)), liblinear_options, 'col');
  
  [a,b,c]=unique(labels_train);
  unique_labels = labels_train(sort(b));
  [~,unique_labels_reverse]=sort(unique_labels);
  svmmodels.unique_labels_reverse = unique_labels_reverse;
 
%   %
%   % perform probability calibration
%   %
%   if isfield(config, 'svm_logisticregression') && config.svm_logisticregression == 1 % use cross validation for this?
%       config.svm_Kernel = 'linear'; % features are already mapped
%       [As Bs]=logisticRegression(hists_train', labels_train, ceil(5*randperm(length(labels_train))/length(labels_train)), @liblinearTrainTest, config);
%       
%       svmmodels.As = As;
%       svmmodels.Bs = Bs;
%   end
end

