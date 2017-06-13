# metadata-extraction-from-lecture-videos
  see papers ["A_multimodal_approach_for_extracting_content_descriptive_metadata_from_lecture_videos"](https://www.researchgate.net/profile/Vidhya_Balasubramanian/publication/274311880_A_multimodal_approach_for_extracting_content_descriptive_metadata_from_lecture_videos/links/561ce64608aea80367266454/A-multimodal-approach-for-extracting-content-descriptive-metadata-from-lecture-videos.pdf)  and  ["MMToC A Multimodal Method for Table of Content Creation in Educational Videos"](http://www.researchgate.net/publication/304417832_MMToC_A_Multimodal_Method_for_Table_of_Content_Creation_in_Educational_Videos)



# A multimodal approach for extracting content descriptive metadata from lecture videos

## 论文摘要：

这篇论文主要包括一个Multimodal Metadata Extraction System 和一个Video Lecture Database System。这里实现了Multimodal Metadata Extraction System。

Multimodal Metadata Extraction System的框架如上图所示，分别从语音和幻灯片上提取n-gram，然后两个独立的贝叶斯分类器将n-gram分类为关键词和非关键词，再根据规则来合并。

## Libraries used:

*   ["Stanford CoreNLP"](https://nlp.stanford.edu/software/)
*   ["mdlp-discretization"](https://github.com/navicto/Discretization-MDLPC)
*   ["pysrt"](https://github.com/byroot/pysrt)
*   ["nltk"](https://github.com/nltk/nltk)
*   ["sklearn"](https://www.baidu.com/link?url=jwc9RTQO2oPgvGY7YDPDKrrZHs3o7oxo_eezrWG78VECamw_wCCTKkttpQuFI55A&wd=&eqid=cef2d2f2000063d70000000659256a78)
*   matplotlib
