function [recRate, confusionMat, scores] = liblinear1vs1Test(hists_test, labels_test, model, config)
%LIBLINEARTEST Summary of this function goes here
%   Detailed explanation goes here

%   addpath([getenv('HOME') '/libs/liblinear-1.91/matlab']);

  if strcmp( config.svm_Kernel , 'linear')

  elseif strcmp( config.svm_Kernel , 'intersection')
      hists_test = vl_homkermap(hists_test', config.homkermap_n, 'kinters', 'gamma', config.homkermap_gamma)' ;
  elseif strcmp( config.svm_Kernel , 'chi-squared')      
      hists_test = vl_homkermap(hists_test', config.homkermap_n, 'kchi2', 'gamma', config.homkermap_gamma)' ;
  else
        error('invalid kernel, kernel %s is not impelemented',config.svm_Kernel);
  end
  
%   if strcmp(class(hists_test),'single')
%       hists_test = double(hists_test);
%   end
  
  if isempty(labels_test)
      labels_test = ones(size(hists_test,1),1);
  end

  
%   %[predicted_label accuracy scores] = predict(labels_test, sparse(hists_test), model, '','row');
%   
%   %avoid output:
%   %[T predicted_label accuracy scores] = evalc('predict(labels_test, sparse(hists_test), model, [],''row'');');
%   results = [];
%   for ii = 1:length(model)
%       ii
%       [T predicted_label accuracy scores] = evalc('predict(double(labels_test), sparse(hists_test), model(ii).model, [],''row'');');
%       results(ii).predicted_label = predicted_label ;
%       results(ii).scores = scores;      
%       results(ii).labels = model(ii).labels;
%   end
%   
%   all_predicted_label = cat(2,results.predicted_label);
%   [b]=hist(all_predicted_label',1:14);
%   [c,d]=max(b);
%   recRate = mean(d==labels_test');
%   scores = results;
%   predicted_label = d;
%   
% 
%   votes_test = scores_test;
%   

  if model.bias == -1
	scores = model.w * hists_test';
  elseif model.bias > 0
	scores = (model.w(:,1:(end-1)) * hists_test') + repmat( model.w(:,end),1,size(hists_test,1));
  else
      error('')
  end
  
    nrClasses = 0.5+sqrt(1/4 + 2*size(model.w,1));
    votes = scores;
    c = 1;
    for ii=1:nrClasses
        for jj=(ii+1):nrClasses
            winIdxs = votes(c,:)>0;
            votes(c,winIdxs) = ii;
            votes(c,~winIdxs) = jj;
            c = c+1;
        end
    end
    voteCounts=hist(votes,1:nrClasses);
    [~,predicted_label]=max(voteCounts);
    recRate = mean(predicted_label' == labels_test);

  
  
%   if length(model.Label) == 2 % two class svm, 
%       scores = [scores -scores];
%   end
  
%   scores = scores';
%   scores = scores(model.unique_labels_reverse,:);

  if isfield(config, 'svm_logisticregression') && config.svm_logisticregression == 1 % use cross validation for this?
      error('not implemented');
%     classes = unique(labels_test);
%     
%     As = model.As;
%     Bs = model.Bs;
%     
%     % assert(length(As) == length(classes));
%     
%     probs = zeros(size(scores));
%     for ii=1:length(model.As)
%         probs(ii,:) = 1./(1+exp(As(ii)*scores(ii,:) + Bs(ii)));
%     end
% 
%     [~, predicted_label] = max(probs) ;
%     scores = probs;
  end
  
  confusionMat = confusionmat(int32(labels_test), int32(predicted_label));
  recRate = sum(labels_test(:) == predicted_label(:)) / length(labels_test);
end

