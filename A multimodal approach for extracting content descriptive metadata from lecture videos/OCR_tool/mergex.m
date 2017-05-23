function [ tmpwords,tmpposition ,tmpwords_position] = mergex( words,position )
    y=zeros(1,size(position,1)-1);
    for i=1:size(position,1)-1
        y(i)=abs(position(i+1,2)-position(i,2));
    end
    %threshold= min(sum(y)/length(y)*1.5,mean(position(:,4)));
    threshold= min(sum(y)/length(y),mean(position(:,4)));
    threshold= max(threshold,mean(position(:,4))*0.3);
    yintervels=find(y>threshold);
    
   
        tmpwords={};
        tmpwords_position={};
        tmpposition=zeros(length(yintervels)+1,4);
        count=1;
        for j=1:length(yintervels)
    %         tmpstr='';
    %         for k=count:yintervels(j)
    %             tmpstr=strcat(tmpstr,strcat(',',words{1,k}));
    %         end
    %        tmpwords{j}=tmpstr(2:end);

            linewords = {}
            linewords_position = []
            for k=count:yintervels(j)
                 linewords{end+1}= words{1,k}
                 linewords_position(end+1,:)= position(k,:)
            end
            tmpwords{end+1} = linewords
            tmpwords_position{end+1}= linewords_position

           % tmpposition(j,4)=max(position(count:yintervels(j),4));
            tmpposition(j,4)=max(position(count:yintervels(j),2)+position(count:yintervels(j),4))-min(position(count:yintervels(j),2));
            tmpposition(j,1)=min(position(count:yintervels(j),1));
            tmpposition(j,2)=min(position(count:yintervels(j),2));
            count=yintervels(j)+1;
            tmpposition(j,3)=position(count-1,1)+position(count-1,3)- tmpposition(j,1);  
        end

    %      tmpstr='';
    %     for k=count:size(position)
    %         tmpstr=strcat(tmpstr,strcat(',',words{1,k}));
    %     end
    %     tmpwords{end+1}=tmpstr(2:end);

        linewords = {}
        linewords_position = []
        for k=count:size(position,1)
             linewords{end+1}= words{1,k}
             linewords_position(end+1,:)= position(k,:)
        end
        tmpwords{end+1} = linewords
        tmpwords_position{end+1}= linewords_position

        tmpposition(length(yintervels)+1,4)=max(position(count:end,2)+position(count:end,4))-min(position(count:end,2));
        tmpposition(length(yintervels)+1,1)=min(position(count:end,1));
        tmpposition(length(yintervels)+1,2)=min(position(count:end,2));
        if length(yintervels) > 0
            count=yintervels(end)+1;
        else
            count = 0
        end
        tmpposition(length(yintervels)+1,3)=position(end,1)+position(end,3)-tmpposition(length(yintervels)+1,1);  
    

end

