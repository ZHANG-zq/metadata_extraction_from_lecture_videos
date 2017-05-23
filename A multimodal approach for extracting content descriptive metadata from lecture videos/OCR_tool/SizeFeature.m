function [scores] = SizeFeature(words,char_position)
    scores = []
        assert(size(words,2) == size(char_position,2))
    for i=1:length(char_position)   
%         letter=isletter(words{i})
%         num=isnumeric(words{i})
%         idx = find(letter | num)
        
        idx =  regexp(words{i},'[0-9a-zA-Z]')
        
        %disp(words{i}(idx))
        [scores(i),min_idx] = min(char_position{i}(idx,4))
        
       %min_idx = idx(min_idx)
       % img=insertShape(img,'rectangle',char_position{i}(min_idx,:),'LineWidth',2,'Color','red');
    end
   % figure;  imshow(img);
    
end

