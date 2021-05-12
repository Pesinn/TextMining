# Text Mining - Articles

Classifies data from news articles to study the landscape of COVID-19 coverage.
A project done in the course [DM882: Text Mining], in the University of Southern Denmark ([SDU]), Odense. Techer: [Konrad Krawczyk].

## Description
With given news articles from multiple sources as .gz [files], each of the following steps are performed:

- Fetch title and description for each article on certain date range
- Text normalization on title and description for each article
- COVID-19 classification based on the normalized text
- Entity recognition to extract all named entities
- ✨Display results (accuracy and propability) ✨

The data is created by [Konrad Krawczyk].

## Technical information
The code is written in [Jupyter notebook] using `Python 3` programming language.

| Plugin | README |
| ------ | ------ |
| NTLK | https://www.nltk.org/ |
| scikit-learn | https://scikit-learn.org/stable/ |

## Run the project

1. Install [Python 3]
2. Install [pip]
3. Install Jupyter notebook
4. Open project folder and run:
```sh
jupyter notebook
```
5. Click TextMining_FinalProject.ipnb
6. Click `Kernel` in the top bar and choose `Restart & Run All`

## Project pipelines
### Fetch data
Article data fetched by date and domain from the `release` folder. Title and description are the important things to consider.
An article object is created:
```json
    {
        "id": "ad5bacf54b7a8ce122ba5fb9077aa1e6",
        "domain": "wsj.com",
        "title": "More Than 1,000 Current and Former CDC Officers Criticize U.S. Covid-19 Response  - WSJ",
        "description": "An open letter criticized the nation’s public-health response to the Covid-19 pandemic and called for the federal agency to play a more central role.",
        "date": "20201017"
    }
```

### Standardization
For each article, we take title and description for standardization processing:
1. Sentence tokenized
    * "I love this project. I hope you also love it" will become: ["I like this project.", "I hope you also like it"]
2. Word tokenization
    * Title will be: ['More', 'Than', '1,000', 'Current', 'and', 'Former', 'CDC', 'Officers', 'Criticize', 'U.S.', 'Covid-19', 'Response', '.']
3. Lemmatization
    * Better -> Good, Rocks -> Rock
4. Stopwords removed
    * Stopwords are fetched from the NLTK library and then filtered from the array. English is only considered but you can add more language to the language functions if you want to consider more languages. 
    ```python
    def languages():
    return [
        "english"
    ]
    ```
5. Punctuations removed
    * .?:!,;‘-’| are removed from the array

NLTK is used for lemmatization (3) and to fetch the stopwords (4). The logic for all other steps (1) (2) (5) is implemented in this project. However, if you set `_useDefault = False`, then NLTK will be used in all steps. This implementation was done to provide the option to compare the final result with- and without NLTK.
Example:
```python
def _sentence_tokenize(text):
    if(_useDefault):
        return _sentence_tokenize_DEFAULT(text, [])
    else:
        return _sentence_tokenize_NLTK(text)
```

Attributes added to the article object:
```json
{
    "title_standardized": "['More', 'Than', '1,000', 'Current', 'Former', 'CDC', 'Officers', 'Criticize', 'U.S.', 'Covid-19', 'Response', '.']",
    "description_standardized": "['An', 'open', 'letter', 'criticized', 'nation', 'public-health', 'response', 'Covid-19', 'pandemic', 'called', 'federal', 'agency', 'play', 'central', 'role']"
}
```
Word stemming is not used because it removes the trailing "e" of many words. Therefore, it affects the named entity analysis that is done later on in this project. As an example, the word "Google" becomes "Googl" and "Apple" becomes "Appl".

### Classification

#### Training- and testing
Articles are split into test- and training sets, where the proportion of test is 20 % and training is 80 %. Each sets are then split into positive and negative, based on keywords extracted from the articles. In this case the keywords used are related to Covid 19, ("covid19", "covid-19", "covid"). Articles that contain any of these words are considered as positive, while all the others are considered as negative.

#### Word frequencies
Next up is to count how many times each word appears in the testing set. An object is created to keep track of that:
```json
{
    "positive": {"word-0", frequency},...,{"word-n", frequency},
    "negative": {"word-0", frequency},...,{"word-n", frequency} 
}
```

#### Prior probability
The prior probability of an article choosen at random from the training set is calculated, P(positive) and P(negative).
```
P(positive) = Positive / All articles
P(negative) = Negative / All articles
```

Also, the probability of each word appearing is calculated and kept in a object.
```json
{
    "positive": {"word-0", probability},...,{"word-n", probability},
    "negative": {"word-0", probability},...,{"word-n", probability} 
}
```

### Entity extraction
All named entities are extracted from each positive article and then counted.
NLTK is used to accomplish this job. More precisely, `pos_tag` and `ne_chunk`.

```python
def _speech_tag_NLTK(wordList):
    return nltk.pos_tag(wordList)

def _named_entities_chunk_NLTK(taggedList):
    return nltk.ne_chunk(taggedList)
```

`pos_tag(wordList)`: Each word is tagged as a noun, conjunction, determiner and so on.
`ne_chunk(taggedList)`: Knows if each word is a named entity after being fed by tagged words

Example of the named entity extraction, where `_speech_tag` and `_named_entities_chunk` encapsulate the NLTK functions mentioned before.
```python
def entity_extraction(wordList):
    extracted = []
    
    # Tagging each word
    tagged = _speech_tag(wordList)
    chunk = _named_entities_chunk(tagged)
    for c in chunk:
        if type(c) == Tree:
            newItems = " ".join([token for token, pos in c.leaves()])
            if newItems not in extracted:
                extracted.append(newItems)
    return extracted
```

This is far from perfect



### Analysis




   [Konrad Krawczyk]: <https://scholar.google.co.uk/citations?user=l-ix1z0AAAAJ&hl=en)>
   [DM882: Text Mining]: <https://odin.sdu.dk/sitecore/index.php?a=searchfagbesk&bbcourseid=N340090101-f-F21&lang=en>
   [SDU]: <https://www.sdu.dk/en>
   [files]: <http://sciride.org/news.html#datacontent>
   [Jupyter notebook]: <https://jupyter.org/>
   [pip]: <https://pip.pypa.io/en/stable/installing/>
   [Python 3]: <https://www.python.org/downloads/>
   
   [PlDb]: <https://github.com/joemccann/dillinger/tree/master/plugins/dropbox/README.md>
   [PlGh]: <https://github.com/joemccann/dillinger/tree/master/plugins/github/README.md>
   [PlGd]: <https://github.com/joemccann/dillinger/tree/master/plugins/googledrive/README.md>
   [PlOd]: <https://github.com/joemccann/dillinger/tree/master/plugins/onedrive/README.md>
   [PlMe]: <https://github.com/joemccann/dillinger/tree/master/plugins/medium/README.md>
   [PlGa]: <https://github.com/RahulHP/dillinger/blob/master/plugins/googleanalytics/README.md>
