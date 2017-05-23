format = 'png'
mode = 'test'

for dirno = 1:1
    dirpath = sprintf('data/%s%d_slide',mode,dirno);
    files=dir(sprintf('%s/*.%s',dirpath,format));
    
    k = length(files);
    
    All_imgpath = {}
    All_words = {}
    All_img = {}
    All_mxwords = {}
    All_words_height = {}
    All_mxwords_position = {} 
    All_mxposition = {}
    min_words_height = 100000
    
    for i = 1:k
        imgpath = sprintf('%s/%s',dirpath,files(i).name);
        if strfind(imgpath,'_test')
            continue;
        end
        img=imread(imgpath);
        img=im2double(img);
 level=graythresh(img); %计算灰度图像的阈值
 img=im2bw(img,level);    %图像二值化
%figure; imshow(img); 
        ocrtext=ocr(img);

        [words,position,char_position]=ocrfilter(ocrtext);
        
        psize = size(position);
        if psize(1) < 1
            continue;
        end
        
        % [ mxwords,mxposition ] = merge( words, char_position )
        [ tmpwords,tmpposition ,tmpwords_position] = mergex( words,position );
        
        All_img{end+1} = img;
        All_imgpath{end+1} = {files(i).name};
        All_words{end+1}   =  words;
        All_mxwords{end+1}   =  tmpwords;
        All_mxposition{end+1}   =  tmpposition;
        All_mxwords_position{end+1}   =  tmpwords_position;
        All_words_height{end+1}   = position(:,4);
        
        min_words_height_t = double(min(position(:,4)));
        if min_words_height > min_words_height_t
            min_words_height = min_words_height_t;
        end
        
    end
    
    %min_LocationScores = min(words_height_vec);
    
    fid_output = fopen( sprintf('%s/slide',dirpath), 'wt','n','UTF-8');
    fprintf(fid_output,'%d\n',size(All_imgpath,2));
    for i = 1:length(All_imgpath)
        Annotation =  im2double(All_img{i});
        
        %fprintf(fid_output,'%s\n',char(All_imgpath{i}));
        %starttime = int64(str2num(char(regexp(char(All_imgpath{i}),'[0-9]{9}','match'))))
        %starttimestamp = sprintf('%02d:%02d:%02d,%03d' ,starttime/(60*60000),mod(starttime,(60*60000))/60000,mod(starttime,(60000))/1000,mod(starttime,1000))
        starttime = char(regexp(char(All_imgpath{i}),'[0-9]{9}','match'))
        fprintf(fid_output,'%s\n',starttime);
        
        idx = 1;
        for j= 1:size(All_mxwords{i},2)
            line = All_mxwords{i}{j};
            line_height = All_mxposition{i}(j,4);
            line_pos = All_mxwords_position{i}{j};
            words_mean_height = mean(All_mxwords_position{i}{j}(:,4));
            for k=1:size(line,2)
                output = sprintf('%s&%d,',char(line{k}),char(words_mean_height));
             
                fprintf(fid_output,output);
                Annotation_text = sprintf('%f',char(words_mean_height));
                Annotation = insertObjectAnnotation(Annotation,'rectangle', line_pos(k,:),Annotation_text,'FontSize',8,'Color','red');
                Annotation=insertShape(Annotation,'rectangle',All_mxposition{i}(j,:),'LineWidth',5,'Color','yellow');
                idx = idx + 1;
            end
            if j == size(All_mxwords{i},2) && k == size(line,2)
                fprintf(fid_output,'.&0'); 
            else
                fprintf(fid_output,'.&0,');
            end
        end
        imwrite( Annotation, strrep(  sprintf('%s/%s',dirpath,char(All_imgpath{i})),  sprintf('.%s',format),  sprintf('_test.%s',format) ) );
        %%imwrite(RGB, strrep( sprintf('%s/%s',dirpath,char(All_imgpath{i})),'.png','_test_height.png'))
        fprintf(fid_output,'\n');
        
    end
    fclose(fid_output);
end