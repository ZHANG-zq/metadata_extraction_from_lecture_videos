# metadata-extraction-from-lecture-videos
  see papers ["A_multimodal_approach_for_extracting_content_descriptive_metadata_from_lecture_videos"](https://www.researchgate.net/profile/Vidhya_Balasubramanian/publication/274311880_A_multimodal_approach_for_extracting_content_descriptive_metadata_from_lecture_videos/links/561ce64608aea80367266454/A-multimodal-approach-for-extracting-content-descriptive-metadata-from-lecture-videos.pdf)  and  ["MMToC A Multimodal Method for Table of Content Creation in Educational Videos"](http://www.researchgate.net/publication/304417832_MMToC_A_Multimodal_Method_for_Table_of_Content_Creation_in_Educational_Videos)



# ["A multimodal approach for extracting content descriptive metadata from lecture videos"](https://www.researchgate.net/profile/Vidhya_Balasubramanian/publication/274311880_A_multimodal_approach_for_extracting_content_descriptive_metadata_from_lecture_videos/links/561ce64608aea80367266454/A-multimodal-approach-for-extracting-content-descriptive-metadata-from-lecture-videos.pdf)

## 论文摘要：

这篇论文主要包括一个Multimodal Metadata Extraction System 和一个Video Lecture Database System。这里实现了Multimodal Metadata Extraction System。

Multimodal Metadata Extraction System的框架如上图所示，分别从语音和幻灯片上提取n-gram，然后两个独立的贝叶斯分类器将n-gram分类为关键词和非关键词，再根据规则来合并。


### 1.文本预处理
关键词抽取的第一步是文本预处理，也就是分别在从语音和幻灯片上获取的文本进行stemming、过滤stopword、POS词性标注、抽取N-gram。

### 2.提取N-gram的语音特征
*   特征一：Adaptive Term Dispersion(ATD)，帮助识别那些在语音上分布平均的keyphrase。
*   特征二：Localspan，帮助识别那些与子话题关联的keyphrase，这些keyphrase一般聚集在某些特定的段落。
*   特征三：C-Value，计算一个phrase与一个document的关联性。
*   特征四：TF-IDF，计算phrase的频率
*   特征五：Cuewords，帮助识别那些跟在‘called as’, ‘defined’之类的词后面的keyphrase

### 3.提取N-gram的幻灯片特征
*   特征一：Mean occurrence ratio，计算每个phrase在整个视频中的平均出现次数（总次数/幻灯片数量）
*   特征二：Contiguous Occurrence Ratio，计算每个phrase的最大连续出现次数除以幻灯片数量，再乘一个系数。
*   特征三：Phrase Height，计算每个phrase高度除以最小phrase高度

### 4.特征值离散化、抽取keyphrase
	由于抽取的很多特征值是连续的实数，贝叶斯分类器不好训练，所以要先将特征值离散到几个区间范围，这篇文章使用了Entropy-MDL方法进行离散化处理。
	然后分别把两个模态得到的特征值训练两个独立的朴素贝叶斯分类器，用这两个分类器可分别得到语音和幻灯片的keyphrase

### 5.多模态抽取的关键词的合并
	从两个模态抽取的关键词按以下规则合并：


## Libraries used:

*   ["Stanford CoreNLP"](https://nlp.stanford.edu/software/)
*   ["mdlp-discretization"](https://github.com/navicto/Discretization-MDLPC)
*   ["pysrt"](https://github.com/byroot/pysrt)
*   ["nltk"](https://github.com/nltk/nltk)
*   ["sklearn"](https://www.baidu.com/link?url=jwc9RTQO2oPgvGY7YDPDKrrZHs3o7oxo_eezrWG78VECamw_wCCTKkttpQuFI55A&wd=&eqid=cef2d2f2000063d70000000659256a78)
*   matplotlib
