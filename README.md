# Combined Neural Topic Modeling-Based and Aspect-Based Sentiment Analysis For Book User Reviews

## Objective
abc
+ Link to previous project
## The Data
abc
+ Original data description
+ Single book data description
### Data Pre-processing
abc
+ Cleaned:
  + Unicode
  + HTML
  + URL
  + Expanded contractions (they're > they are)
  + Emoticons
  + Excess white space
  + Force lowercase
  + Split reviews into individual sentences
+ Cleaned using Python libraries:
  + SpaCy
  + re
  + contractions
  + bs4
  + unicode
## Aspect Extraction
abc
+ Seed keywords
  + Setting
  + Character
  + Plot
  + Conflict
  + Theme
  + Point-of-view
  + Tone
  + Style
  + Dialogue
  + Action
  + Description
  + Exposition
  + Motivation
  + Symbolism
  + Climax
  + Resolution
  + Imagery
  + Pacing
  + Writing
  + Author 
+ POS Patterns
  + Adjective > noun(s)
  + Adverb > adverb > noun(s)
  + Adverb > adjective > noun(s)
  + Adverb > verb > noun(s)
  + Verb > noun(s)
+ KeyphraseCountVectorizer
  + KeyphraseCountVectorizer is an enhanced version of the CountVectorizer which is designed to find key phrases using part-of-speech patterns
  + Stopwords
+ KeyBERT model
  + KeyBERT utilizes BERT embeddings to produce keywords/phrases that best represent a document
  + https://maartengr.github.io/KeyBERT/index.html
  + distilbert-base-nli-mean-tokens
+ Ended up with matched lists of nouns and their descriptive words
## Neural Topic Modeling
+ Guided topic modeling for seed keywords using BERTopic
+ https://maartengr.github.io/BERTopic/getting_started/hierarchicaltopics/hierarchicaltopics.html#tf-idf
+ Sentence Transformer embedding model to create numerical representations of sentences
+ https://huggingface.co/sentence-transformers
+ Sentence Transformer model all-MiniLM-L6-v2 comes recommended with good out-of-the-box performance
## Sentiment Analysis
+ Pre-trained siebert/sentiment-roberta-large-english sentiment model
  + https://huggingface.co/siebert/sentiment-roberta-large-english?text=I+like+you.+I+love+you
### Aspect-Based Sentiment Analysis
+ Produces sentiments of aspects that may be buried within a sentence
+ Most sentiment analyses produce a macro-level sentiment for a single word, sentence, paragraph, document, etc.
+ Allows you to determine sentiment for specific nouns within sentences
+ In this case, it is for the elements of writing and books from the seed keyword list
+ Sentiment scores are 0-1 with POSITIVE OR NEGATIVE tags
+ Reversed polarity for NEGATIVE tags (*= -1)
+ Lemmatized the nouns
+ Filtered to aspects from the seed keyword list
+ Aggregated mean sentiment score and count of aspect mentions in reviews
+ Final product is plot
  + Color indicates mean sentiment (-1/red most negative, 1/blue most positive)
  + X-axis represents the number of times the aspect was mentioned by reviewers
+ The elements of writing and books provided by the seed keywords give decent metrics that can be observed for most books
+ This plot can inform a potentional consumer regarding meta-information on how others liked the writing/author and also gives an idea of the weight of those sentiments (bar length/color saturation)
### Neural Topic Modeling-Based Sentiment Analysis
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
