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
For each article, title and description is concatinated into a single string before standardization processing:
1. Sentence cleaned with repetitive information from outlets
    * Some outlets add additional text as postfix of their titles. That text is deleted from the title in the first step. See the list below:
    ```python
    text_to_remove = ["- ABC News", "| Al Jazeera", "| US & Canada News", "- BBC Sport", "- BBC Reel", "- BBC News", "| CBC News", "- Washington Times", "| Reuters", "- The New York Times", "| Euronews", "| Daily Mail", "- CBS News"]
    ```
2. Sentence tokenized
    * "I love this project. I hope you also love it" will become: ["I like this project.", "I hope you also like it."]
3. Word tokenization
    * Tokenize each word in the text. ["I", "like", "this", "project", ".", "I", "hope", "you", "also", "like", "it", "."]
4. Lemmatization
    * Better -> Good, Rocks -> Rock
5. Stopwords removed
    * Stopwords are fetched from the NLTK library and then filtered from the array. English is only considered but you can add more language to the language functions if you want to consider more languages. 
    ```python
    def languages():
    return [
        "english"
    ]
    ```
6. Punctuations removed
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

Attribute added to the article object:
```json
{
    "standardized": "['More', 'Than', '1,000', 'Current', 'Former', 'CDC', 'Officers', 'Criticize', 'U.S.', 'Covid-19', 'Response', '.', 'An', 'open', 'letter', 'criticized', 'nation', 'public-health', 'response', 'Covid-19', 'pandemic', 'called', 'federal', 'agency', 'play', 'central', 'role']",
}
```
Word stemming is not used because it removes the trailing "e" of many words. Therefore, it affects the named entity analysis that is done later on in this project. As an example, the word "Google" becomes "Googl" and "Apple" becomes "Appl".

### Classification

#### Training- and testing
Articles are split into test- and training sets, where the proportion of test is 20 % and training is 80 %. Each sets are then split into positive and negative, based on keywords extracted from the articles. In this case the keywords used are related to Covid 19, ("covid19", "covid-19", "covid", "coronavirus"). Articles that contain any of these words are considered as positive, while all the others are considered as negative.

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

Also, the probability of each word appearing is calculated and kept in a object: `word_probability`.
```json
{
    "positive": {"word-0", probability},...,{"word-n", probability},
    "negative": {"word-0", probability},...,{"word-n", probability} 
}
```

### Entity extraction
All named entities are extracted from each positive article and then counted. NLTK is used to accomplish this job. More precisely, `pos_tag` and `ne_chunk`.

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
The code aims to combine named entities that should be combined. See an usage example below:
```python
print(entity_extraction(['Australia', 'fire', 'Smoke', 'turn', 'New', 'Zealand', "'", 'yellow']))
['Australia', 'New Zealand']
```
Here we see how Austalia is considered as a single entity, while New Zealand has been combined into a single entity also, which makes perfect sense.

However, this is far from perfect. Here is an exampe where this logic breaks:
```python
print(entity_extraction(['Vale', 'Withheld', 'Information', 'From', 'Regulator', 'Before', 'Brazil', 'Dam', 'Disaster', '- WSJ']))
['Vale', 'Withheld Information', 'Brazil Dam Disaster']
```
Here there word Brazil has been combined with `Dam Disaster`, which is not what we want. 

### Analysis
The probability of each word in all articles from the testing set is now being multiplied and put into new object. The probability of each word is kept in the object mentioned above called `word_probability`. 
```python
P(words|positive) = P(positive) * P("all") * P("tokenized") * P("words") * P("appearing") * P("in") * P("article") * P("related") * P("to") * P("covid-19")
P(words|negative) = P(negative) * P("all") * P("tokenized") * P("words") * P("appearing") * P("in") * P("article") * P("unrelated") * P("to") * P("covid-19")
```
Based on this information, predictive values are calculated - `True Positive (TP)`, `True Negative (TN)`, `False Positive (FP)`, `False Negative (TP)`
It's done by comparing the probability of all `P(words|positive)` to `P(words|negative)` in the whole test set, for each article.
Then, the accuracy is calculated using the formula
```
(TP + TN) / (TP + TN + FP + FN)
```
Last, all positive test sets and negative test sets are counted.

### Display
The information displayed for each run:
* Number of positive articles
* Number of negative articles
* Date range
* List of all outlets included in the process
* Accuracy
* Portion of positive articles
* A graph that shows x-most popular entities

```python
    print("========")
    print(f"Finished processing articles:")
    print(f"Positive: {positive}")
    print(f"Negative: {negative}")    
    print(f"DateFrom: {dateFrom}")
    print(f"DateTo: {dateTo}")
    print(f"Domains: {domains}")
    print(f"Accuracy: {displayAccuracy} %")
    print(f"Portion of positive: {displayPositive} % ")
    display_graph(f"{_namedEntities} most popular named entities from {dateFrom} to {dateTo}", x_axis, y_axis)
    print("========")
```
### Run the application
This is how the application is executed to estimate the propostion of articles on COVID-19
* As proportion of all articles in 2020
* As proportion of all articles in each month of 2020
* As proportion of articles from CNN in 2020
```python
# Process all articles from 2020
text_mine_articles(get_domains(), "20200101", "20201231")

# Process all articles for each month in 2020
each_month("2020")

# Process all artiles in an outlet in 2020
text_mine_articles(["bbc.com"], "20200101", "20201231")
```
Only a portion of the whole dataset is used in this project.
These are the outlets considered:
* 9news.com.au
* abc.net.au
* abcnews.go.com
* aljazeera.com
* bbc.com
* cbc.ca
* cbsnews.com
* dailymail.co.uk
* euronews.com
* nytimes.com
* reuters.com
* washingtontimes.com

### Results
Here are the results from running the application using the code in `Run the application section`. If you have dark background in Github, you won't see the metrics.
![Entities_all](https://i.imgur.com/m8JZWp1.png)

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
