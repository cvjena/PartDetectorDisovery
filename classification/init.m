% add all files to path, initialize libraries

% add all dirs
t = dir('.');
for ii=1:length(t)
%      t(ii).name
     if t(ii).isdir & strcmp(t(ii).name,'3rdparty')
         % do nothing, add when needed?
%          'excluded'
     elseif t(ii).isdir & t(ii).name(1) ~= '.'
         addpath(genpath([pwd '/' t(ii).name]))
     end
end

clear t ii

% add libs
sets = settings();


addpath(sets.libraries_liblinear);
