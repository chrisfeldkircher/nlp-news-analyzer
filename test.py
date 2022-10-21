import text_analyzer as ta

urls = ta.UrlLoader.loadFromFile('urls6.data')
'''
scrapper = ta.Webscrapper(urls=urls)
scrapper.getText()

articles = scrapper.articles
'''

nlp = ta.NLP(urls=urls)
#predictions = nlp.predict()
#articles = nlp.articles
#pred = nlp.cleanPred
#nlp.summarize()
nlp.foward()
cleanedText = nlp.cleanedText

def saveText(texts, name):
    with open(f'{name}.txt', 'w') as f:
        for i in range(len(texts)):
            for j in range(len(texts[i])):
                f.write(texts[i][j])
                f.write('\n')
        f.write('\n')
        f.write('\n')
        f.write('\n')

saveText(cleanedText, 'article')