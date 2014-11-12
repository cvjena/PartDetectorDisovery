function [config,confDiffString] = parseConfigs(configDefault, conf)
    config = configDefault;
    confDiffString = '';
    if exist('conf','var')
		for field = sort(fieldnames(conf))'
            if ~isfield(config,field{1}) || ~compare(config.(field{1}),conf.(field{1}))
                % add to string
                value = conf.(field{1});
                if isnumeric(value)
                    value = num2str(value);
                    value(strfind(value,' ')) = '-';
                elseif iscell(value)
                    value = value{:};
                    if iscell(value)
                        value=value{:};
                    end
                end
                confDiffString  = [confDiffString  '_c-' field{1} '-' value];
            end
			config.(field{1}) = conf.(field{1}); % why is {1} necessary?
		end
	end
end



function res = compare(a,b)

    if ~strcmp(class(a),class(b)) 
        assert('different types...')
    end

    res = false;
    if ischar(a)
        res = strcmp(a,b);
    elseif isnumeric(a)
        if isscalar(a) && isscalar(b)
            res = a == b;
        else
            if all(size(a) == size(b))
                res = all(a == b);
            else
                res = false;
            end
        end
    elseif iscell(a)
        if all(size(a) == size(b))
            res = false;
        else
            partRes = [];
            if (length(a)~=length(b))
                res=false;
                return;
            end
            for ii = 1:length(a)
                partRes(ii) = compare(a{ii},b{ii});
            end
            res = all(partRes);
        end
    else
        assert(['type not implemented: ' class(a)])
    end

end
