function [A,B] = logisticRegression(hists, labels, split, functionTrainTest, config)
%logisticRegression 

    classes = unique(labels);
    scores = zeros(length(classes),length(labels));
    
    splitIds = unique(split);
    
    
    for ii = 1:length(splitIds)
        currentSplitId = splitIds(ii);
        currentSplit = (split == currentSplitId);
        
        hists_train = hists(~currentSplit, :);
        hists_test = hists(currentSplit, :);
        labels_train = labels(~currentSplit);
        labels_test = labels(currentSplit);
        
        config.svm_logisticregression = 0;
        [~, ~, currentScores ] = functionTrainTest(hists_train, labels_train, hists_test, labels_test, config);
        
        scores(:,currentSplit) = currentScores;
    end

    %probs = zeros(size(scores));
    for ii=1:length(classes)
        bin_labels = labels == classes(ii);
        [A(ii),B(ii)] = platt(scores(ii,:), bin_labels);
        
        %probs(:,ii) = 1./(1+exp(A(ii)*scores(:,ii) + B(ii)));
    end

    %[drop, imageEstClassProbs] = max(probs, [], 2) ;
    %[drop, imageEstClassScores] = max(scores, [], 2) ;

end