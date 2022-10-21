import text_analyzer as ta
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='text-analyzer')
    parser.add_argument('-c', '--conf', help='Define the path of the configuration-file - wrap it around " marks')
    parser.add_argument('-u', '--urls', help='Define the name of the file where the urls are stored in the url directory - wrap it around " marks')
    parser.add_argument('-sd', '--saveData', help='Save websites content to a file after processing')
    parser.add_argument('-df', '--dataFile', help='Define the name of the website-content-file in the Text_Data dirctory - wrap it around " marks')
    parser.add_argument('-sp', '--savePrediction', help='Save predictions to a file after processing')
    parser.add_argument('-pf', '--predictionFile', help='Define the name of the Prediction-file in the Pred_Data directory - wrap it around " marks')
    parser.add_argument('-db', '--dataBase', help='Load urls from DataBase connection defined in the conf-file')
    parser.add_argument('-v', '--visualize', help='Visualize frequency')
    args = parser.parse_args() #parser for the arguments

    if not (args.urls or args.dataFile):
        parser.error('No action requested, add -u (define url-file path) or -df (define website-content file)')

    if args.urls and args.dataFile:
        parser.error('Can not load url-files and data-files at the same time. please only provide one of the options: -u (define url-file path to load from) or -df (define website-content file to load from)')

    if args.saveData and args.dataFile:
        parser.error('Can not save a data-file and load data-files at the same time. please only provide one of the options: -sd (save websites content to a file) or -df (define website-content file to load from)')

    if args.savePrediction and args.predictionFile:
        parser.error('Can not save a pred-file and load pred-files at the same time. please only provide one of the options: -sp (save Predictions content to a file) or -pf (define Prediction-file to load from)')

    if args.urls:
        urls = ta.UrlLoader.loadFromFile(args.urls)
        if args.conf:
            if args.saveData: 
                if args.predictionFile: 
                    if args.visualize: nlp = ta.NLP(urls=urls, saveData=True, loadPred=True, loadName=args.predictionFile, conf=args.conf, showfreq=True)
                    else: nlp = ta.NLP(urls=urls, saveData=True, loadPred=True, loadName=args.predictionFile, conf=args.conf)
                else: 
                    if args.savePrediction: 
                        if args.visualize: nlp = ta.NLP(urls=urls, saveData=True, loadPred=False, conf=args.conf, savePred=True, showfreq=True)
                        else: nlp = ta.NLP(urls=urls, saveData=True, loadPred=False, conf=args.conf, savePred=True)
                    else: 
                        if args.visualize: nlp = ta.NLP(urls=urls, saveData=True, loadPred=False, conf=args.conf, showfreq=True)
                        else: nlp = ta.NLP(urls=urls, saveData=True, loadPred=False, conf=args.conf)
            else: 
                if args.predictionFile: 
                    if args.visualize: nlp = ta.NLP(urls=urls, saveData=False, loadPred=True, loadName=args.predictionFile, conf=args.conf, showfreq=True)
                    else: nlp = ta.NLP(urls=urls, saveData=False, loadPred=True, loadName=args.predictionFile, conf=args.conf)
                else: 
                    if args.savePrediction: 
                        if args.visualize: nlp = ta.NLP(urls=urls, saveData=False, loadPred=False, conf=args.conf, savePred=True, showfreq=True)
                        else: nlp = ta.NLP(urls=urls, saveData=False, loadPred=False, conf=args.conf, savePred=True)   
                    else: 
                        if args.visualize: nlp = ta.NLP(urls=urls, saveData=False, loadPred=False, conf=args.conf, showfreq=True)   
                        else: nlp = ta.NLP(urls=urls, saveData=False, loadPred=False, conf=args.conf)   
        else:
            if args.saveData: 
                if args.predictionFile: 
                    if args.visualize: nlp = ta.NLP(urls=urls, saveData=True, loadPred=True, loadName=args.predictionFile, conf=args.conf, showfreq=True)
                    else: nlp = ta.NLP(urls=urls, saveData=True, loadPred=True, loadName=args.predictionFile, conf=args.conf)
                else: 
                    if args.savePrediction: 
                        if args.visualize: nlp = ta.NLP(urls=urls, saveData=True, loadPred=False, conf=args.conf, savePred=True, showfreq=True)
                        else: nlp = ta.NLP(urls=urls, saveData=True, loadPred=False, conf=args.conf, savePred=True)
                    else: 
                        if args.visualize: nlp = ta.NLP(urls=urls, saveData=True, loadPred=False, conf=args.conf, showfreq=True)
                        else: nlp = ta.NLP(urls=urls, saveData=True, loadPred=False, conf=args.conf)
            else: 
                if args.predictionFile: 
                    if args.visualize: nlp = ta.NLP(urls=urls, saveData=False, loadPred=True, loadName=args.predictionFile, conf=args.conf, showfreq=True)
                    else: nlp = ta.NLP(urls=urls, saveData=False, loadPred=True, loadName=args.predictionFile, conf=args.conf)
                else: 
                    if args.savePrediction: 
                        if args.visualize: nlp = ta.NLP(urls=urls, saveData=False, loadPred=False, conf=args.conf, savePred=True, showfreq=True)
                        else: nlp = ta.NLP(urls=urls, saveData=False, loadPred=False, conf=args.conf, savePred=True)  
                    else: 
                        if args.visualize: nlp = ta.NLP(urls=urls, saveData=False, loadPred=False, conf=args.conf, showfreq=True)  
                        else: nlp = ta.NLP(urls=urls, saveData=False, loadPred=False, conf=args.conf)  

        nlp.foward() 
    
    if args.dataFile:
        if args.conf:
            if args.predictionFile: 
                if args.visualize: nlp = ta.NLP(loadData= True, name = args.dataFile, loadPred=True, loadName=args.predictionFile, conf=args.conf, showfreq=True)
                else: nlp = ta.NLP(loadData= True, name = args.dataFile, loadPred=True, loadName=args.predictionFile, conf=args.conf)
            else: 
                if args.savePrediction: 
                    if args.visualize: nlp = ta.NLP(loadData= True, name = args.dataFile, conf=args.conf, savePred=True, showfreq=True)
                    else: nlp = ta.NLP(loadData= True, name = args.dataFile, conf=args.conf, savePred=True)
                else: 
                    if args.visualize: nlp = ta.NLP(loadData= True, name = args.dataFile, conf=args.conf, showfreq=True)
                    else: nlp = ta.NLP(loadData= True, name = args.dataFile, conf=args.conf)
        else:
            if args.predictionFile: 
                if args.visualize: nlp = ta.NLP(loadData= True, name = args.dataFile, loadPred=True, loadName=args.predictionFile, showfreq=True)
                else: nlp = ta.NLP(loadData= True, name = args.dataFile, loadPred=True, loadName=args.predictionFile)
            else: 
                if args.savePrediction: 
                    if args.visualize: nlp = ta.NLP(loadData= True, name = args.dataFile, savePred=True, showfreq=True)
                    else: nlp = ta.NLP(loadData= True, name = args.dataFile, savePred=True)
                else: 
                    if args.visualize: nlp = ta.NLP(loadData= True, name = args.dataFile, showfreq=True)
                    else: nlp = ta.NLP(loadData= True, name = args.dataFile)

        nlp.foward()

    if args.dataBase:
        raise NotImplementedError("This Feature is not yet Implemented!")
        if args.conf:
            connection = ta.DBConnection(args.conf)
            connection.connect()
            urls = connection.getTableData()
        else:
            connection = ta.DBConnection()
            connection.connect()
            urls = connection.getTableData()

        if args.conf:
            if args.saveData: 
                if args.predictionFile: nlp = ta.NLP(urls=urls, saveData=True, loadPred=True, loadName=args.predictionFile, conf=args.conf)
                else: nlp = ta.NLP(urls=urls, saveData=True, loadPred=False, conf=args.conf)
            else: 
                if args.predictionFile: nlp = ta.NLP(urls=urls, saveData=False, loadPred=True, loadName=args.predictionFile, conf=args.conf)
                else: 
                    if args.savePrediction: ta.NLP(urls=urls, saveData=False, loadPred=False, conf=args.conf, savePred=True)   
                    else: nlp = ta.NLP(urls=urls, saveData=False, loadPred=False, conf=args.conf)   
        else:
            if args.saveData: 
                if args.predictionFile: nlp = ta.NLP(urls=urls, saveData=True, loadPred=True, loadName=args.predictionFile, conf=args.conf)
                else: 
                    if args.savePrediction: ta.NLP(urls=urls, saveData=True, loadPred=False, conf=args.conf, savePred=True)   
                    else: nlp = ta.NLP(urls=urls, saveData=True, loadPred=False, conf=args.conf)
            else: 
                if args.predictionFile: nlp = ta.NLP(urls=urls, saveData=False, loadPred=True, loadName=args.predictionFile, conf=args.conf)
                else: 
                    if args.savePrediction: nlp = ta.NLP(urls=urls, saveData=False, loadPred=False, conf=args.conf, savePred=True)   
                    else: nlp = ta.NLP(urls=urls, saveData=False, loadPred=False, conf=args.conf)   

        nlp.foward()