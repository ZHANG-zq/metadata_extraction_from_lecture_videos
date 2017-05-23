function [ mxwords,mxposition ] = merge( words,position )
    y=zeros(1,size(position,1)-1);
    for i=1:size(position,1)
        y(i)=position(i,2)+position(i,4)/2;
    end
    yintervals=[];
    intervals=[];
    for i=1:size(position)-1
        intervals(end+1)=y(i+1)-y(i);
        if intervals(i)>=10
            yintervals(end+1)=i;
        end
    end
    count=1;
    mergeinfo={};
    for i=1:size(yintervals,2)
        tmp=[];
        while count<=yintervals(i)
            tmp(end+1)=count;
            count=count+1;
        end
        mergeinfo{end+1}=tmp;
    end
    tmp=[];
    while count<=size(position,1)
        tmp(end+1)=count;
        count=count+1;
    end
    mergeinfo{end+1}=tmp;
    mxwords={};
    mxposition=zeros(length(mergeinfo),4);
    for i=1:length(mergeinfo)
        tmpwords='';
        for j=mergeinfo{1,i}(1):mergeinfo{1,i}(end)
            tmpwords=strcat(tmpwords,words{1,j});
            tmpwords=strcat(tmpwords,',');
        end
        tmpwords=tmpwords(1:end-1);
        mxwords{1,i}=tmpwords;
        mxposition(i,4)=max(position(mergeinfo{1,i}(1):mergeinfo{1,i}(end),2)+position(mergeinfo{1,i}(1):mergeinfo{1,i}(end),4))-min(position(mergeinfo{1,i}(1):mergeinfo{1,i}(end),2));
        mxposition(i,1)=min(position(mergeinfo{1,i}(1):mergeinfo{1,i}(end),1));
        mxposition(i,2)=min(position(mergeinfo{1,i}(1):mergeinfo{1,i}(end),2));
        mxposition(i,3)=position(mergeinfo{1,i}(end),1)+position(mergeinfo{1,i}(end),3)- mxposition(i,1);  
    end
    

end

