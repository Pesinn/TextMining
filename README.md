# Text Mining - Articles

Classifies data from news articles to study the landscape of COVID-19 coverage.
A project done in the course [DM882: Text Mining], in the University of Southern Denmark ([SDU]), Odense. Techer: [Konrad Krawczyk].

## Description
With given news articles from multiple sources as .gz files, each of the following steps are performed:

- Fetch title and description for each article on certain date range
- Text normalization on title and description for each article
- COVID-19 classification based on the normalized text
- Entity recognition to extract all named entities
- Display results (accuracy and probability)

The data is created by [Konrad Krawczyk]. More information about it can be found [here].

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
### Fetch- and standardize data
Article data fetched by date and domain from the `release` folder. Title and description are the important things to consider.

Here is an example from one of the articles
```json
    "title": "More Than 1,000 Current and Former CDC Officers Criticize U.S. Covid-19 Response  - WSJ",
    "description": "An open letter criticized the nation’s public-health response to the Covid-19 pandemic and called"
```

An article object is created where both title and description have been combined and standardized using code described in the standardization section below:
```json
    {
        "id": "ad5bacf54b7a8ce122ba5fb9077aa1e6",
        "domain": "wsj.com",
        "standardized": "['More', 'Than', '1,000', 'Current', 'Former', 'CDC', 'Officers', 'Criticize', 'U.S.', 'Covid-19',
        'Response', 'An', 'open', 'letter', 'criticized', 'nation', 'public-health', 'response', 'Covid-19', 'pandemic', 'called']",
        "date": "20201017"
    }
```

### Standardization
For each article, title and description is concatinated into a single string before standardization processing:

1. Sentence cleaned with repetitive information from outlets
    -  Some outlets add additional text as postfix of their titles, often included after the last "-" or "|" character in a sentence. The purpose of this code is to get rid of that.
    ```python
    # Remove trailing text like "- ABC News" or "| Al Jazeera"
    def split_and_remove_last_part(text, split):
        spl = text.split(split)
        
        # At least one "split char" is included in text
        if(len(spl) > 1):
            spl.pop()
            
        return ' '.join(map(str, spl))

    def clean_text(text):
        text = split_and_remove_last_part(text, " - ")
        text = split_and_remove_last_part(text, " | ")
    return text
    ```
2. Sentence tokenization
    - "I like this project. I hope you also like it" will become: ["I like this project.", "I hope you also like it."]
3. Word tokenization
    - Tokenize each word in the text. ["I", "like", "this", "project", ".", "I", "hope", "you", "also", "like", "it", "."]
4. Lemmatization
    - Finds the root form of the word: Better -> Good, Rocks -> Rock
5. Stopwords removed
    - Stopwords are fetched from the NLTK library and then filtered from the array. English is only considered in this case. However, more language can be added to the language function to get stopwords from other languages.
    ```python
    def languages():
        return [
            "english"
        ]
    ```
6. Punctuations removed
    - .?:!,;‘-’| are removed from the tokenized array

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
    "standardized": "['More', 'Than', '1,000', 'Current', 'Former', 'CDC', 'Officers', 'Criticize', 'U.S.', 'Covid-19',
    'Response', 'An', 'open', 'letter', 'criticized', 'nation', 'public-health', 'response', 'Covid-19',
    'pandemic', 'called']",
}
```
Word stemming is not used because it removes the trailing "e" of many words. Therefore, it affects the named entity analysis that is done later on in this project. As an example, the word "Google" becomes "Googl" and "Apple" becomes "Appl".

### Classification

#### Training- and testing
Articles are split into test- and training sets, where the proportion of test is 20 % and training is 80 %. Each sets are then split into positive and negative, based on keywords extracted from the articles. In this case the keywords used are related to Covid 19, ("covid19", "covid-19", "covid", "coronavirus"). Articles that contain any of these words are considered as positive, while all the others are considered as negative. The function `train_test_split` from the sklean library is used for splitting the data.

#### Word frequencies
Next up is to count how many times each word appears in the testing set. An object is created to keep track of that:
```json
{
    "positive": [{"word-0": "frequency"}, {"word-1": "frequency"}],
    "negative": [{"word-0": "frequency"}, {"word-1": "frequency"}]
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
    "positive": [{"word-0": "probability"}, {"word-n": "probability"}],
    "negative": [{"word-0": "probability"}, {"word-n": "probability"}]
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
P(words|positive) = P(positive) * P("all") * P("tokenized") * P("words") * P("appearing") * P("in")
* P("article") * P("related") * P("to") * P("covid-19")
P(words|negative) = P(negative) * P("all") * P("tokenized") * P("words") * P("appearing") * P("in")
* P("article") * P("unrelated") * P("to") * P("covid-19")
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
1. Number of positive articles
2. Number of negative articles
3. Number of negative articles
4. Date range
5. List of all outlets included in the process
6. Accuracy
7. Portion of positive articles
8. A graph that shows x-most popular entities

## Run the application
This is how the application is executed to estimate the propostion of articles on COVID-19
* As proportion of all articles in 2020
* As proportion of all articles in each month of 2020
* As proportion of articles from CNN in 2020

```python
# Process all articles for each month in 2020
each_month("2020")

# Process all artiles in an outlet in 2020
process_articles(["bbc.com"], "20200101", "20201231")
```
Only the outlets that publishes articles in English are used in this project.
These are the outlets considered:

9news.com.au, morningstaronline.co.uk, abc.net.au, abcnews.go.com, afr.com, aljazeera.com, apnews.com, bbc.com, bostonglobe.com, breitbart.com, businessinsider.com, cbc.ca, cbsnews.com channel4.com,thesun.co.uk, chicagotribune.com, cnbc.com, csmonitor.com, ctvnews.ca, dailymail.co.uk, dailystar.co.uk, dw.com, economist.com, edition.cnn.com, euronews.com, express.co.uk, france24.com, newsweek.com, globalnews.ca, huffpost.com, independent.co.uk, independent.ie, inquirer.com, irishexaminer.com, irishmirror.ie, irishtimes.com, itv.com, latimes.com, liverpoolecho.co.uk, macleans.ca, metro.co.uk, mirror.co.uk, montrealgazette.com, msnbc.com, nbcnews.com, news.com.au, news.sky.com, news.yahoo.com, newshub.co.nz, npr.org, nypost.com, nytimes.com, nzherald.co.nz, politico.com, reuters.com, rnz.co.nz, rt.com, rte.ie, sbs.com.au, scoop.co.nz, scotsman.com, slate.com, smh.com.au, standard.co.uk, stuff.co.nz, telegraph.co.uk, theage.com.au, theatlantic.com, theglobeandmail.com, theguardian.com, thehill.com, thejournal.ie, thestar.com, thesun.ie, thetimes.co.uk, thewest.com.au, time.com, torontosun.com, upi.com,foxnews.com, usatoday.com, vancouversun.com, walesonline.co.uk, washingtonpost.com, washingtontimes.com, westernjournal.com, wnd.com, wsj.com

## Results
Here are the results from running the application on the English articles. However, there is no data for December 2020. If you have dark background in Github, you won't see the metrics.

### January 2020
* Domains: All
* Positive: 5517
* Negative: 185474
* Accuracy: 99.10 %
* Portion of positive: 2.89 %
![Entities_all_jan_2020](https://i.imgur.com/1M5ejoW.png)*

### February 2020
* Domains: All
* Positive: 19490
* Negative: 164756
* Accuracy: 97.36 %
* Portion of positive: 10.58 %
![Entities_all_feb_2020](https://i.imgur.com/f0AnKWe.png)

### March 2020
* Domains: All
* Positive: 132157
* Negative: 72225
* Accuracy: 78.56 %
* Portion of positive: 64.66 %
![Entities_all_mar_2020](https://i.imgur.com/90C1xV8.png)

### April 2020
* Domains: All
* Positive: 133457
* Negative: 61320
* Accuracy: 76.47 %
* Portion of positive: 68.52 %
![Entities_all_apr_2020](https://i.imgur.com/xTLrGV1.png)

### May 2020
* Domains: All
* Positive: 82867
* Negative: 112645
* Accuracy: 85.50 %
* Portion of positive: 42.38 %
![Entities_all_may_2020](https://i.imgur.com/OADiPom.png)

### June 2020
* Domains: All
* Positive: 40278
* Negative: 156586
* Accuracy: 92.70 %
* Portion of positive: 20.46 %
![Entities_all_jun_2020](https://i.imgur.com/qrQKQb1.png)

### July 2020
* Domains: All
* Positive: 44929
* Negative: 151213
* Accuracy: 92.29 %
* Portion of positive: 22.91 %
![Entities_all_jul_2020](https://i.imgur.com/1vP2tyu.png)

### August 2020
* Domains: All
* Positive: 32600
* Negative: 148989
* Accuracy: 93.66 %
* Portion of positive: 17.95 %
![Entities_all_aug_2020](https://i.imgur.com/suzakPm.png)

### September 2020
* Domains: All
* Positive: 27700
* Negative: 144673
* Accuracy: 94.07 %
* Portion of positive: 16.07 %
![Entities_all_sep_2020](https://i.imgur.com/xUxdwl0.png)

### October 2020
* Domains: All
* Positive: 22387
* Negative: 68384
* Accuracy: 91.85 %
* Portion of positive: 24.66 %
![Entities_all_oct_2020](https://i.imgur.com/P6oVnli.png)

### November 2020
* Domains: All
* Positive: 59
* Negative: 451
* Accuracy: 92.75 %
* Portion of positive: 11.57 %
![Entities_all_nov_2020](https://i.imgur.com/9ZbaTZm.png)

### December 2020
* No data

### Whole year 2020
* Domains: All
* Positive: 541441
* Negative: 1266716
* Accuracy: 90.39 %
* Portion of positive: 29.94 %
![Positives_all_pos_2020](https://i.imgur.com/HA7xyGj.png)
![Entities_all_pos_2020](https://i.imgur.com/oje3yUA.png)

### Whole year 2020 - BBC
* Domain: bbc.com
* Positive: 5371
* Negative: 12045
* Accuracy: 92.09 %
* Portion of positive: 30.84 %
![Entities_bbc_entities_2020](https://i.imgur.com/IHqiugL.png)



   [Konrad Krawczyk]: <https://scholar.google.co.uk/citations?user=l-ix1z0AAAAJ&hl=en)>
   [DM882: Text Mining]: <https://odin.sdu.dk/sitecore/index.php?a=searchfagbesk&bbcourseid=N340090101-f-F21&lang=en>
   [SDU]: <https://www.sdu.dk/en>
   [here]: <http://sciride.org/news.html#datacontent>
   [Jupyter notebook]: <https://jupyter.org/>
   [pip]: <https://pip.pypa.io/en/stable/installing/>
   [Python 3]: <https://www.python.org/downloads/>
   
   [PlDb]: <https://github.com/joemccann/dillinger/tree/master/plugins/dropbox/README.md>
   [PlGh]: <https://github.com/joemccann/dillinger/tree/master/plugins/github/README.md>
   [PlGd]: <https://github.com/joemccann/dillinger/tree/master/plugins/googledrive/README.md>
   [PlOd]: <https://github.com/joemccann/dillinger/tree/master/plugins/onedrive/README.md>
   [PlMe]: <https://github.com/joemccann/dillinger/tree/master/plugins/medium/README.md>
   [PlGa]: <https://github.com/RahulHP/dillinger/blob/master/plugins/googleanalytics/README.md>
