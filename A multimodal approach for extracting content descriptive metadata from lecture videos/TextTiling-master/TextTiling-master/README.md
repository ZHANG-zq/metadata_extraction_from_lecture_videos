Given a text presented in ‘alien-life.txt’, find all the possible segment boundaries using TextTiling
algorithm. Reference boundaries are marked with “$$” in the text. Perform the following tasks. (Any
programming language may be used. However, you should be able to execute the programs.)


1. Remove all the punctuations and lowercase the characters.
2. Remove function words (link to list of function words is provided )
3. Perform stemming (you may use NLTK for this)
4. Implement TextTilingt without using NLTK libray  and segment “alien-life.txt” with it. Use (m-sigma) to be the threshold where is the (m)mean depth score and (sigma)is the standard deviation.
5. Implement Windowdiff measure and report segmentation performance
6. Vary pseudo sentence length from 10 to 100 and plot Windowdiff value. Report optimalpseudo sentence length.
