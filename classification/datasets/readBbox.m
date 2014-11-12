function [bbox] = readBbox(imageName, config)
%READBOX Reads the gt bbox from file
%   Reads the gt bbox from file. Dataset is determined from path.

    setts = settings();
    
%     if ~exist(imageName,'file')
%         imageName = [getenv('HOME') '/' imageName];
%     end

    % dataset specific

    if ~isempty(strfind(imageName,'CUB_200_2011'))
        dataBaseDir = setts.dataset_cub200_2011;    
        fid = fopen([ dataBaseDir '/images.txt' ]);
        images = textscan(fid, '%s %s');
        fclose(fid);
        slashPos = strfind(imageName,'/');
        fname = imageName(( slashPos(end-1) + 1):end );

        imageId = find(strcmp(images{2},fname));
        assert(length(imageId) == 1);

        if (exist([setts.cachedir '/bboxes_cache.mat'],'file'))
            bboxes = load([setts.cachedir '/bboxes_cache.mat']);
            bboxes = bboxes.bboxes;
        else
            bboxes = load([ dataBaseDir '/bounding_boxes.txt' ]);
            save([setts.cachedir '/bboxes_cache.mat'],'bboxes');
        end

        bbox.left = bboxes(imageId,2);
        bbox.top = bboxes(imageId,3);
        bbox.right = bbox.left + bboxes(imageId,4);
        bbox.bottom= bbox.top + bboxes(imageId,5);
    else
%         warning('there were no bounding boxes for this image. Is that supposed to happen?')
        im = imread(imageName);
        bbox.left = 1;
        bbox.top = 1;
        bbox.right = size(im,2);
        bbox.bottom = size(im,1);
    end
end

% ----- Subfunction PARSECHILDNODES -----
function children = parseChildNodes(theNode)
% Recurse over node children.
children = [];
if theNode.hasChildNodes
   childNodes = theNode.getChildNodes;
   numChildNodes = childNodes.getLength;
   allocCell = cell(1, numChildNodes);

   children = struct(             ...
      'Name', allocCell, 'Attributes', allocCell,    ...
      'Data', allocCell, 'Children', allocCell);

    for count = 1:numChildNodes
        theChild = childNodes.item(count-1);
        children(count) = makeStructFromNode(theChild);
    end
end
end

% ----- Subfunction MAKESTRUCTFROMNODE -----
function nodeStruct = makeStructFromNode(theNode)
% Create structure of node info.

nodeStruct = struct(                        ...
   'Name', char(theNode.getNodeName),       ...
   'Attributes', parseAttributes(theNode),  ...
   'Data', '',                              ...
   'Children', parseChildNodes(theNode));

if any(strcmp(methods(theNode), 'getData'))
   nodeStruct.Data = char(theNode.getData); 
else
   nodeStruct.Data = '';
end
end

% ----- Subfunction PARSEATTRIBUTES -----
function attributes = parseAttributes(theNode)
% Create attributes structure.

attributes = [];
if theNode.hasAttributes
   theAttributes = theNode.getAttributes;
   numAttributes = theAttributes.getLength;
   allocCell = cell(1, numAttributes);
   attributes = struct('Name', allocCell, 'Value', ...
                       allocCell);

   for count = 1:numAttributes
      attrib = theAttributes.item(count-1);
      attributes(count).Name = char(attrib.getName);
      attributes(count).Value = char(attrib.getValue);
   end
end
end

