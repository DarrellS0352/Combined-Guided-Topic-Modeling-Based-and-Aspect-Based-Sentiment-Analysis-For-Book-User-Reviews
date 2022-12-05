# Combined Guided Topic Modeling-Based and Aspect-Based Sentiment Analysis For Book User Reviews

## Objective
The purpose of this project is to give additional insights about recommended books to a user utilizing topic modeling, aspect extraction, and sentiment analysis. It is intended to be an extension of a previous project that provides book recommendations with neural networks which can be found at [this repository](https://github.com/DarrellS0352/Neural-Collaborative-Filtering-Book-Recommendation-System).
## The Data
The data used is Amazon product review data gathered by Ni et al. (2019). There is raw review data, ratings only data, data subsets for products that have at least five reviews, and per-category data available for each product category. The raw data has book metadata (2,935,525 books) and the book review data (51,311,621 reviews). The two datasets are joined on the book ID to add book titles to the reviews. To meet computing constraints, the data is then filtered to the top 10 genres with the most unique books and books that have 50 or more reviews. The final dataset is approximately 13 million reviews for 64,000 books and 5.3 million users. For development and presentation considerations, the data was filtered to a single book with 1,539 reviews.
### Data Pre-processing
The raw review texts were pre-processed with the following methods:
+ Unicode removal
+ HTML removal
+ URL removal
+ Emoticon removal
+ Expanded contractions (they're > they are)
+ Forced lowercase
+ Split reviews into individual sentences
## Aspect Extraction
Aspect extraction was done using KeyBERT and some custom wrangling methods. [KeyBERT](https://maartengr.github.io/KeyBERT/index.html) utilizes BERT embeddings to produce keywords/phrases that best represent a document. It is used in combination with KeyphraseCountVectorizer which is an enhanced version of the CountVectorizer and is designed to find key phrases using part-of-speech patterns and also takes care of stopwords. The part-of-speech patterns used are variants from the work of Banjar et al. (2020). It parses syntactic dependencies pulling out phrases such as adjective > noun or verb > noun. The model is fed a list of seed keywords which guide the model (if possible) towards phrases related to said list. The seed keywords are various words related to aspects of writing and books. The words used are setting, character, plot, conflict, theme, point-of-view, tone, style, dialogue, action, description, exposition, motivation, symbolism, climax, resolution, imagery, pacing, writing, and author. The output is matched lists of nouns and their descriptive words
## Guided Topic Modeling
A [sentence transformer embedding model](https://huggingface.co/sentence-transformers) is used to create numerical representations of the cleaned sentences. The sentence embeddings are then given to the topic model. The topic modeling is done with [BERTopic](https://maartengr.github.io/BERTopic/index.html). The guided topic modeling functionality is utilized by also feeding the list of seed keywords to this model. This determines the high-level topics written about in the reviews using transformers and class-based term frequency inverse document frequency (TF-IDF) to create dense topic clusters. Each topic produced consists of the top four keywords of the topic concatenated with underscores (ex: windows_drive_dos_file). The topics are assigned a topic number with -1 being outliers that don't fit with others well enough.
## Sentiment Analysis
The pre-trained sentiment classification model used was the [siebert/sentiment-roberta-large-english sentiment model](https://huggingface.co/siebert/sentiment-roberta-large-english?text=I+like+you.+I+love+you). The sentiment model was applied to the key phrase for aspect-based sentiment analysis. For the topic modeling-based sentiment analysis it was applied to the cleaned sentence which is represented by the assigned topic.
 
### Aspect-Based Sentiment Analysis
Aspect-based sentiment analysis extracts aspects (nouns) and their descriptors (adjectives, verbs, etc.) and applies sentiments to the extracted words at a sub-sentence level. Most sentiment analyses produce a macro-level sentiment for a single word, sentence, paragraph, document, etc. In this case, it is the sentiments in book reviews for the elements of writing and books from the seed keyword list.

The sentiment model gives 0-1 scores with POSITIVE or NEGATIVE labels. The scores with NEGATIVE labels are then reversed in polarity and the nouns are lemmatized. Given the output includes every noun that met the part-of-speech patterns, the output is filtered to just the words from the seed keyword list. Then the output is aggregated to determine the mean sentiment score and count of mentions for each word. The final product is a bar graph with the color indicating the mean sentiment and the x-axis portraying the aspect counts. The elements of writing and books provided by the seed keywords give decent metrics that can be used to evaluate most books. This plot can inform a potentional consumer regarding meta-information on how others liked the writing/author and also gives an idea of the weight of those sentiments (bar length/color saturation).
### Guided Topic Modeling-Based Sentiment Analysis
+ Provides meta and content focused sentiments
  + Possible to have topics similar to previously shown aspects but also produced content-based topics
  + For example:
    + The topic 'scifi_recommend_asimove_readable' gives the impression the book is comparable to Isaac Asimov novels
    + The topic 'holden_miller_detective_idealistic' is describing two primary characters and someone unfamiliar with the story may not completely understand
+ The same wrangling techniques were applied except the original cleaned sentences were fed through the sentiment model
  + This is acceptable because the sentences are represented by assigned topic
## References
Banjar, A., Ahmed, Z., Daud, A., Abbasi R.A., Dawood, H. (2020). Aspect-Based Sentiment Analysis for Polarity Estimation of Customer Reviews on Twitter. Computers, Materials & Continua 2021, 67(2), 2203-2225. https://doi.org/10.32604/cmc.2021.014226 

Ni, J., Li, J., McAuley, J. (2019). Justifying Recommendations using Distantly-Labeled Reviews and Fine-Grained Aspects. Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), 188–197. http://dx.doi.org/10.18653/v1/D19-1018
