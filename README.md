# Anime Rankings Using Aspect Based Sentiment Analysis

This project explores whether anime rankings can be improved using Aspect-Based Sentiment Analysis (ABSA), fuzzy logic, and Word2Vec. Review data for five anime titles were scraped from MyAnimeList (MAL), and aspect terms were extracted and classified for sentiment using ATE and ATSC. Word2Vec was used to group similar terms into five canonical categories—Narrative, Visual, Audio, Character, and Miscellaneous—which were then weighted and combined using fuzzy logic to produce an overall rating. Compared to MAL’s official scores, the ABSA-based ratings more accurately reflected the sentiments expressed in user reviews, offering a more nuanced, data-driven approach to anime evaluation.

## Developer Setup 
1. Download GoogleNews-vectors-negative300.bin.gz from https://code.google.com/archive/p/word2vec/ and put it in the pretrained_models folder. ([direct link](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g))
2. Run ```pip3 install -r requirements.txt```
3. Run anime_absa.ipynb to generate output for ABSA ratings of chosen anime from MyAnimeList. Just copy and pasted url link in  the form of the previous ones. 

## Acknowledgements
https://github.com/kevinscaria/InstructABSA