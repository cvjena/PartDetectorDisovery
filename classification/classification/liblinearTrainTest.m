function [ recRate, confusionMat, scores, liblinearModel ] = liblinearTrainTest( hists_train, labels_train, hists_test, labels_test, config )
%LIBLINEARTRAINTEST Compute recognition rates using liblinear svm

    % normalize features for liblinear
    % subtract mean
%     mean_val=mean(hists_train,1);
%     hists_train=hists_train-repmat(mean_val,size(hists_train,1),1);
%     hists_test=hists_test-repmat(mean_val,size(hists_test,1),1);
    
%     % normalize rows
%     hists_train=hists_train./repmat(sum(hists_train,2),1,size(hists_train,2));
%     hists_test =hists_test ./repmat(sum(hists_test ,2),1,size(hists_test ,2));
    

    liblinearModel = liblinearTrain(hists_train, labels_train, config);
    [recRate, confusionMat, scores] = liblinearTest(hists_test, labels_test, liblinearModel, config);
end

