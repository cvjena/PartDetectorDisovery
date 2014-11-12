function [recRate, confusionMat, scores] = liblinearTest(hists_test, labels_test, model, config)
%LIBLINEARTEST Summary of this function goes here
%   Detailed explanation goes here

%   addpath([getenv('HOME') '/libs/liblinear-1.91/matlab']);

  if strcmp( config.svm_Kernel , 'linear')
      hists_test = hists_test';
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
  
%   [predicted_label,accuracy,scores] = predict(double(labels_test), sparse(double(hists_test)), model, 'col');
%   scores=scores';
  %avoid output:
  %[T predicted_label accuracy scores] = evalc('predict(labels_test, sparse(hists_test), model, [],''row'');');
%   [T predicted_label accuracy scores] = evalc('predict(double(labels_test), sparse(hists_test), model, [],''row'');');
%   [T predicted_label accuracy scores] = evalc('predict(double(labels_test), sparse(hists_test), model, [],''row'');');
  
  if model.bias == -1
	scores = model.w * hists_test;
  elseif model.bias > 0
	scores = (model.w(:,1:(end-1)) * hists_test) + repmat( model.w(:,end),1,size(hists_test,2));
  else
      error('')
  end
  
  if length(model.Label) == 2 % two class svm, 
      scores = [scores; -scores];
  end
  
  scores = scores(model.unique_labels_reverse,:);
  
  [~,predicted_label] = max(scores);

  if isfield(config, 'svm_logisticregression') && config.svm_logisticregression == 1 % use cross validation for this?
    classes = unique(labels_test);
    
    As = model.As;
    Bs = model.Bs;
    
    % assert(length(As) == length(classes));
    
    probs = zeros(size(scores));
    for ii=1:length(model.As)
        probs(ii,:) = 1./(1+exp(As(ii)*scores(ii,:) + Bs(ii)));
    end

    [~, predicted_label] = max(probs) ;
    scores = probs;
  end
  
  confusionMat = confusionmat(int32(labels_test), int32(predicted_label));
  recRate = sum(labels_test(:) == predicted_label(:)) / length(labels_test);
end

