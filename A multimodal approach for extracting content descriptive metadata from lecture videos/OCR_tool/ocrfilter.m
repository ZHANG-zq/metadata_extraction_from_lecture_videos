function [ words,position,char_position,confidence ] = ocrfilter( ocrtext )
    words={};
    position=[];
    confidence = []
    char_i=1;
    char_position={};
    count=0;
    for i=1:length(ocrtext.Words)
        if ocrtext.WordConfidences(i)>=0.8
%                char=find(ischar(ocrtext.Words{i})>0);
%                letter=find(isletter(ocrtext.Words{i})>0);
 %              if length(char)>0 & length(letter)>0 & length(strtrim(ocrtext.Words{i}))>0
 
              %ocrtext.Words{i} = regexprep(ocrtext.Words{i},'[:#]',' ')
              if length(regexp(ocrtext.Words{i},'[0-9a-zA-Z]'))>=1
               
                    count=count+1;
                    %words{end+1}  = strtrim(regexprep(strrep(ocrtext.Words{i},',',''),'[^\w]',' '))  %%最好根据strim掉的区域相应地调整position，不然boldness feature有问题。
                    %words{end+1} = regexprep(ocrtext.Words{i},'[^\w]',' ')   %没有改变word的长度，char_position也就能够对应了，算sizefeatrue就方便
                    
                    %words{end+1} = regexprep(ocrtext.Words{i},'[!"#%&()*+,./:;<=>?@\[\\\]^_`{|}~]',' ') 
                    words{end+1} = regexprep(ocrtext.Words{i},'[!"#%&()*+,./:;<=>?@\[\\\]^_`{|}~]',' ') 
                    assert( length(words{end}) == length(ocrtext.Words{i}) )
                    
                    position(count,:)=ocrtext.WordBoundingBoxes(i,:);
                    confidence(end+1 ) =ocrtext.WordConfidences(i ); 
                    
                    p =  strfind( ocrtext.Text(char_i:end),ocrtext.Words{i})
                    char_position{end+1} = []
                    for j = 1:length(words{end})
                        char_position{end}(j ,:)  = ocrtext.CharacterBoundingBoxes( char_i +p(1) +j-2,:)
                        %disp(ocrtext.Text(char_i +p(1) +j-2))
                    end
                    assert( length(words{end}) == size(char_position{end},1) )
                    char_i = char_i+p(1)+length(ocrtext.Words{i})
                    
              end 
        else
            char_i = char_i+length(ocrtext.Words{i})
        end
    end
    % char_position = ocrtext.CharacterBoundingBoxes
end

