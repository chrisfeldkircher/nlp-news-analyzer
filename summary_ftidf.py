import text_analyzer as ta
import re
import math

urls = ta.UrlLoader.loadFromFile('urls4.data')
'''
scrapper = ta.Webscrapper(urls=urls)
scrapper.getText()

articles = scrapper.articles
'''

nlp = ta.NLP(urls=urls)
predictions = nlp.predict()
articles = nlp.articles
pred = nlp.cleanPred
#nlp.summarize()
#nlp.foward()

def cleanUpText(articles):
    cleanedText = []

    for article in articles:
        temp = []
        for data in article[0]['data']:
            for i in range(len(data)):
                clean_sent = re.sub(r'[^\w\s\-\_\'\§\$\%\&\=\#\<\>]','',list(data[i].values())[0])
                temp.append(clean_sent)
        cleanedText.append(temp)
    
    return cleanedText

cleanedText = cleanUpText(articles)

def createBoW(articles):
    bow = []
    wordSet = []
    for article in articles:
        temp = []
        set_of_words = {}
        for sentence in article:
            temp.append(sentence.split(' '))
            set_of_words = set(set_of_words).union(set(sentence.split(' ')))
        bow.append(temp)
        wordSet.append(set_of_words)
    
    return bow, wordSet

bow, wordSet = createBoW(cleanedText)

def createDicts(bow, wordSet):
    wordDicts = []
    for i in range(len(wordSet)):
        temp = []
        for sentence in bow[i]:
            wordDict = dict.fromkeys(wordSet[i], 0)

            for word in sentence:
                wordDict[word] += 1
        
            temp.append(wordDict)

        wordDicts.append(temp)
    
    return wordDicts

wordDicts = createDicts(bow, wordSet)

def calcTF(wordDicts):
    tfDict = []

    for article in wordDicts:
        temp = []
        for sentence in article:
            tempDict = {}
            for word, count in sentence.items():
                tempDict[word] = count/ sum(sentence.values())
            
            temp.append(tempDict)
        
        tfDict.append(temp)

    return tfDict

tfDict = calcTF(wordDicts)

def calcIDF(wordDicts):
    idfDict = []

    for article in wordDicts:
        tempDict = dict.fromkeys(article[0].keys(), 0)
        for sentence in article:
            for word, num in sentence.items():
                if num > 0:
                    tempDict[word] += 1

        for word, num in tempDict.items():
            tempDict[word] = math.log(len(article) / float(num))
    
        idfDict.append(tempDict)

    return idfDict
    
idfDict = calcIDF(wordDicts)

def calcTFIDF(tfDict, idfDict):
    tfidf = []
    for i in range(len(tfDict)):
        temp = []
        for sentence in tfDict[i]:
            tempDict = {}
            for word, val in sentence.items():
                tempDict[word] = val * idfDict[i][word]
            
            temp.append(tempDict)
        tfidf.append(temp)
    return tfidf

tfidf = calcTFIDF(tfDict, idfDict)

