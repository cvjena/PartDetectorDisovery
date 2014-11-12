function sets = settings()
%SETTINGS Global settings like pathes to libraries and datasets

    % path to liblinear
    sets.libraries_liblinear = ['/path/to/liblinear/matlab']; 
    % path to the virtual env containing the modified DeCAF framework
    sets.python_env = ['../python_virtual_env/'];
    sets.libraries_decaf = '../decaf-tools/';
    
    % path to CUB200-2011 dataset
    sets.dataset_cub200_2011 = ['/path/to/CUB_200_2011/'];
    sets.outputdir = ['out'];
    sets.cachedir =  ['tmp'];
end
