function [ recRate, confusionMat, scores, liblinearModel ] = liblinear1vs1TrainTest( hists_train, labels_train, hists_test, labels_test, config )
%LIBLINEARTRAINTEST Compute recognition rates using liblinear svm

    liblinearModel = liblinear1vs1Train(hists_train, labels_train, config);
    [recRate, confusionMat, scores] = liblinear1vs1Test(hists_test, labels_test, liblinearModel, config);
end

