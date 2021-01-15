from django.shortcuts import render
import urllib.request
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse, Http404

from io import BytesIO
import os
import datetime
from django.http import JsonResponse

import pandas
import librosa
import numpy
import joblib
from sklearn import preprocessing
from sklearn.svm import SVC
from music_model.Utilities import extract_features
from snownlp import SnowNLP #文本分析
from collections import Counter
import jieba
import jieba.posseg as psg
import re

Genres = ['Blues', 'Classical', 'Country', 'Disco', 'HipHop',
          'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']
#Types = ["神的愛", "禱告", "讚美", "相交相愛", "仰賴神", "盼望應許", "默想神恩", "奉獻"]
Types2 = ["讚美", "爭戰", "宣告", "醫治", "仰望", "尋求", "信靠", "悔改", "感恩", "奉獻"]
ImgList = ["jesus-4779545_1920.jpg","jesus-4779546_1920.jpg","jesus-4779544_1920.jpg","jesus-4779549_1920.jpg","jesus-4779542_1920.jpg","jesus-3754861_1280.png","jesus-4779545_1920.jpg","jesus-5382512_1920.jpg","jesus-4929681_1920.jpg","jesus-4779548_1920.jpg"]
ImgRecList = ["平靜.jpg","鄉村.jpg","FIGHT_FOR_LOVE.jpg","500x500.jpg","勵志.jpg","images.png"]
praise = "祂,他,偉大,慶賀,揚聲,歡呼,讚美,快樂,歌頌,稱頌,奇妙,歌唱,喜樂,全知,全能,興起,哈利路,榮耀,跳舞,拍手,跳躍,歡欣,喜悅"
thanksgiving = "你,祢,禰,獻上,感恩,感謝,真好,在一起,降臨,感謝,謝謝,預備,春雨,恩友,全然,真諦,約定"
worship = "我,耶和華,敬拜,切慕,同在,異象,永活,寶座,聖潔,羔羊,賜我,聖名,平安,屈膝,降臨,配得,尊榮,觸動,尊崇,尊貴,榮耀,仰望,恩典,渴慕"
feekback = "決定,需要,禮物,贏得,真誠,愛你,相愛,燃燒,面前,復興,獻上,賜福,宣教,注目,耶穌,愛你,相聚,看見,復興,閃亮"
delKeyWord = ["魔鏡歌詞網"]

# Name of Features in CSV File
Feature_Names = ['meanSpecCentroid', 'stdSpecCentroid',
                 'meanSpecBandwidth', 'stdSpecBandwidth',
                    'meanSpecContrast', 'stdSpecContrast',
                    'meanSpecRollof', 'stdSpecRollof',
                    'meanSpecFlux', 'stdSpecFlux',
                    'meanZCR', 'stdZCR',
                    'meanTempo', 'stdTempo',
                    'meanMFCC_01', 'stdMFCC_01',
                    'meanMFCC_02', 'stdMFCC_02',
                    'meanMFCC_03', 'stdMFCC_03',
                    'meanMFCC_04', 'stdMFCC_04',
                    'meanMFCC_05', 'stdMFCC_05',
                    'meanMFCC_06', 'stdMFCC_06',
                    'meanMFCC_07', 'stdMFCC_07',
                    'meanMFCC_08', 'stdMFCC_08',
                    'meanMFCC_09', 'stdMFCC_09',
                    'meanMFCC_10', 'stdMFCC_10',
                    'meanMFCC_11', 'stdMFCC_11',
                    'meanMFCC_12', 'stdMFCC_12',
                    'meanMFCC_13', 'stdMFCC_13',
                    'meanChromaSTFT_01', 'stdChromaSTFT_01',
                    'meanChromaSTFT_02', 'stdChromaSTFT_02',
                    'meanChromaSTFT_03', 'stdChromaSTFT_03',
                    'meanChromaSTFT_04', 'stdChromaSTFT_04',
                    'meanChromaSTFT_05', 'stdChromaSTFT_05',
                    'meanChromaSTFT_06', 'stdChromaSTFT_06',
                    'meanChromaSTFT_07', 'stdChromaSTFT_07',
                    'meanChromaSTFT_08', 'stdChromaSTFT_08',
                    'meanChromaSTFT_09', 'stdChromaSTFT_09',
                    'meanChromaSTFT_10', 'stdChromaSTFT_10',
                    'meanChromaSTFT_11', 'stdChromaSTFT_11',
                    'meanChromaSTFT_12', 'stdChromaSTFT_12',
                    'meanMelScale_01', 'stdMelScale_01',
                    'meanMelScale_02', 'stdMelScale_02',
                    'meanMelScale_03', 'stdMelScale03',
                    'meanMelScale_04', 'stdMelScale_04',
                    'meanMelScale_05', 'stdMelScale_05',
                    'meanMelScale_06', 'stdMelScale_06',
                    'meanMelScale_07', 'stdMelScale_07',
                    'meanMelScale_08', 'stdMelScale_08',
                    'meanMelScale_09', 'stdMelScale_09',
                    'meanMelScale_10', 'stdMelScale_10',
                    'meanTonnetz_01', 'stdTonnetz_01',
                    'meanTonnetz_02', 'stdTonnetz_02',
                    'meanTonnetz_03', 'stdTonnetz_03',
                    'meanTonnetz_04', 'stdTonnetz_04',
                    'meanTonnetz_05', 'stdTonnetz_05',
                    'meanTonnetz_06', 'stdTonnetz_06'
                 ]

datasetsForFit = numpy.load('music_model/classification_dataset.npy',allow_pickle = True)
datasetsGenreForFit = numpy.load('music_model/genre_dataset.npy',allow_pickle = True)

@csrf_exempt
def my_api(request):

    try:
        # 上傳過來的檔案存放於記憶體中
        # <class 'django.core.files.uploadedfile.InMemoryUploadedFile'>
        music = request.FILES["upload_music"]

        if 'lyrics' not in request.POST:
            lyrics = ""
        else:
            lyrics = request.POST["lyrics"].strip()


        fs = FileSystemStorage()
        file_path = fs.save('app_music_classification_api/static/media/song.mp3', music)
        #print(file_path)
        song, sr = librosa.load(file_path, sr=22050, duration=5.0)

        # Extract Features of Test Files and Save Them in Array
        data_features = numpy.array(extract_features(song))

        if(len(lyrics) != 0):
            lyrics = re.sub('[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>﹥?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]','',lyrics)
            # 歌詞關鍵字統計
            newLyrics,sentiments,psgDict = nlpProcess(lyrics) #nlp分析
            pCount, tCount, wCount, fCount = count_keyword(newLyrics)

            #詞性數量輸入(比例)
            psgKey = list(psgDict.keys())
            psgVal = []
            psgCount = 0
            for i in psgKey:
                psgVal.append(len(psgDict[i])) 
                psgCount = psgCount + len(psgDict[i])
            for i in range(len(psgVal)):
                psgVal[i] = psgVal[i]/psgCount

        else:
            pCount, tCount, wCount, fCount = 0, 0, 0, 0
            sentiments = -1


        genreType = int(get_genre([data_features]))
        musicData = pandas.DataFrame(
            [data_features], columns=Feature_Names)  # 基本歌曲特徵
        musicData["genre"] = genreType  # 取得曲風
        musicData["praiseKeyWord"] = pCount
        musicData["thanksgivingKeyWord"] = tCount
        musicData["worshipKeyWord"] = wCount
        musicData["feekbackKeyWord"] = fCount
        musicData["sentiments"] = sentiments
        for i in range(len(psgVal)):
            musicData[psgKey[i]] = psgVal[i]

        # musicData.to_csv("show_upload_data.csv", mode="a",
        #                       index=False, header=True, encoding='utf_8_sig')

        jsonData = {}
        jsonData['song_name'] = request.POST["song_name"].strip()
        jsonData['type'] = get_music_type(musicData)
        jsonData['genre'] = Genres[genreType]
        jsonData['singer'] = request.POST["singer"].strip()
        jsonData['bpm'] = str(data_features[12])
        jsonData['tone'] = get_likely_tone(data_features)
        jsonData['media_url'] = request.POST["media_url"].strip()
        file_path = file_path.replace("app_music_classification_api","http://163.18.42.232:8000")
        jsonData['path'] = file_path
        jsonData['nlp_psg'] = psgDict
        jsonData['status'] = 'Done'
        print(jsonData)
        response = JsonResponse(jsonData)
        response['Access-Control-Allow-Origin'] = "*"

        return response

    except:
        response = JsonResponse({"Status": "Error"})
        response['Access-Control-Allow-Origin'] = "*"

        return response


@csrf_exempt
def new_train_data(request):

    try:
        dataset_numpy = None
        music = request.FILES["upload_music"]
        fs = FileSystemStorage()
        file_path = fs.save('media/train_music/song.mp3', music)

        signal, sr = librosa.load(file_path, sr=22050, duration=5.0)
        dataset_numpy = numpy.array(extract_features(signal))

        dataset_pandas = pandas.DataFrame(
            [dataset_numpy], columns=Feature_Names)  # 基本歌曲特徵
        dataset_pandas["genre"] = get_genre([dataset_numpy])  # 取得曲風

        lyrics = request.POST["lyrics"].strip()
        lyrics = re.sub('[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>﹥?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]','',lyrics)
        # 歌詞關鍵字統計
        newLyrics,sentiments,psgDict = nlpProcess(lyrics) #nlp分析

        pCount, tCount, wCount, fCount = count_keyword(newLyrics)
        dataset_pandas["praiseKeyWord"] = pCount
        dataset_pandas["thanksgivingKeyWord"] = tCount
        dataset_pandas["worshipKeyWord"] = wCount
        dataset_pandas["feekbackKeyWord"] = fCount
        dataset_pandas["sentiments"] = sentiments

        #詞性數量輸入(比例)
        psgKey = list(psgDict.keys())
        psgVal = []
        psgCount = 0
        for i in psgKey:
            psgVal.append(len(psgDict[i])) 
            psgCount = psgCount + len(psgDict[i])
        for i in range(len(psgVal)):
            dataset_pandas[psgKey[i]] = psgVal[i]/psgCount

        # 新增客製類型
        dataset_pandas["type"] = Types2[int(request.POST["type_value"])]
        dataset_pandas["file_path"] = file_path

        # 新增歌名
        dataset_pandas["song_name"] = request.POST["song_name"].strip()

        # 新增時間
        now = datetime.datetime.now()
        dataset_pandas["upload_time"] = now.strftime(
            "%Y-%m-%d %H:%M:%S")  # 轉換為指定的格式
        dataset_pandas["lyrics"] = lyrics  # 歌詞(往後可能更新)

        dataset_pandas.to_csv("orig_datasets.csv", mode="a",
                            index=False, header=False, encoding='utf_8_sig')

        response = JsonResponse({"Status": "Done"})
        response['Access-Control-Allow-Origin'] = "*"

        return response
    except:
        response = JsonResponse({"Status": "Error"})
        response['Access-Control-Allow-Origin'] = "*"

        return response

@csrf_exempt
def get_types(request):
    jsonData = {}
    count = 0

    for i in Types2:
        jsonData[str(count)] = i
        count = count + 1

    response = JsonResponse(jsonData, json_dumps_params={
                            'ensure_ascii': False})
    response['Access-Control-Allow-Origin'] = "*"

    return response


@csrf_exempt
def types_img(request):
    jsonData = {}
    
    for i in range(len(Types2)):
        jsonData[str(i)] = {"info":Types2[i], "image":"http://163.18.42.232:8000/static/img/"+ImgList[i]}

    response = JsonResponse(jsonData)
    response['Access-Control-Allow-Origin'] = "*"
    return response


@csrf_exempt
def songlist_img(request):
    jsonData = {}
    
    for i in range(len(ImgRecList)):
        jsonData[str(i)] = {"info":"推薦歌單 "+str(i+1), "image":"http://163.18.42.232:8000/static/img/"+ImgRecList[i]}

    response = JsonResponse(jsonData)
    response['Access-Control-Allow-Origin'] = "*"
    return response


@csrf_exempt
def dataset_download(request):

    file_path = "orig_datasets.csv"
    jsonData = {}
    count = 0

    if os.path.exists(file_path):

        df=pandas.read_csv(file_path)
        for index, row in df.iterrows():
            d=row.to_dict()
            jsonData[str(count)] = d
            count = count + 1
            
        response = JsonResponse(jsonData, json_dumps_params={
                        'ensure_ascii': False})
        response['Access-Control-Allow-Origin'] = "*"

        return response

    raise Http404


def get_genre(data):
    # Scale ALL Variables Between -1 to 1
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    # fit transform on training data
    scaler.fit(datasetsGenreForFit)
    # transform training data
    data_X = scaler.transform(data)
    #print(data_X)

    # Predict Genres
    svm = joblib.load('music_model/genre_model.pkl')
    predicts = svm.predict(data_X)


    print(predicts)
    return str(Genres.index(str(predicts[0])))


def get_likely_tone(data):

    song_chroma = []
    # pitches in 12 tone equal temperament
    pitches = ['C', 'C#','D','D#','E','F','F#','G','G#','A','A#','B']

    for i in range(12):
        song_chroma.append(data[40+i*2])

    # print note to value relations
    for y in range(len(song_chroma)):
        print(str(pitches[y]) + '\t' + str(song_chroma[y]))

    # select the most dominate pitch
    pitch_id = song_chroma.index(max(song_chroma))
    pitch = pitches[pitch_id]

    min_third_id = (pitch_id+3) %12
    maj_third_id = (pitch_id+4) %12

    # check if the musical 3rd is major or minor
    if song_chroma[min_third_id] < song_chroma[maj_third_id]:
        third = 'major'
        return(str.format('{},{}', pitch, third))
    elif song_chroma[min_third_id] > song_chroma[maj_third_id]:
        third = 'minor'
        return(str.format('{},{}', pitch, third))
    else:
        return(str.format('{}', pitch))


def count_keyword(text):

    pCount = 0
    tCount = 0
    wCount = 0
    fCount = 0

    #取不重複的結果
    for p in praise.split(","):
        if(text.find(p) != -1):
            pCount = pCount + 1
    for t in thanksgiving.split(","):
        if(text.find(t) != -1):
            tCount = tCount + 1
    for w in worship.split(","):
        if(text.find(w) != -1):
            wCount = wCount + 1
    for f in feekback.split(","):
        if(text.find(f) != -1):
            fCount = fCount + 1

    #取總數，最後要比例就好
    count = pCount+tCount+wCount+fCount

    return  pCount/count, tCount/count,wCount/count,fCount/count


def nlpProcess(text):

    result = []
    tempDel = []
    psgRes = {"nList":[],"vList":[],"aList":[],"anList":[],"vnList":[],"nsList":[]}

    sen = Counter(SnowNLP(text).sentences)
    noRepSen = list(sen.keys())
    

    for i in range(len(noRepSen)): #刪除不要的關鍵字段落
        for j in delKeyWord:
            if (noRepSen[i].find(j) != -1 or noRepSen[i].strip()==""):
               tempDel.append(i)
    for i in tempDel: 
        del noRepSen[i]

    newText = ""
    for i in noRepSen: #組合新的Text
        newText = newText + "，" + i

    sentimentsRes = SnowNLP(newText).sentiments #分析文字情感

    #將詞性分析儲存到字典
    for i in jieba.lcut(newText):
        for word, flag in psg.cut(i):
            if(flag == "n"):
                if(not word in psgRes["nList"]):
                    psgRes["nList"].append(word)
            elif(flag == "v"):
                if(not word in psgRes["vList"]):
                    psgRes["vList"].append(word)
            elif(flag == "a"):
                if(not word in psgRes["aList"]):
                    psgRes["aList"].append(word)
            elif(flag == "an"):
                if(not word in psgRes["anList"]):
                    psgRes["anList"].append(word)
            elif(flag == "vn"):
                if(not word in psgRes["vnList"]):
                    psgRes["vnList"].append(word)
            elif(flag == "ns"):
                if(not word in psgRes["nsList"]):
                    psgRes["nsList"].append(word)

    return newText,sentimentsRes,psgRes


def get_music_type(data):

    scaler = preprocessing.MaxAbsScaler()
    # fit transform on training data
    scaler.fit(datasetsForFit)
    # transform training data
    data_X = scaler.transform(data)
    #print(data_X)

    # Predict Genres
    svm = joblib.load('music_model/model_music_classification.pkl')
    predicts = svm.predict(data_X)

    return str(predicts[0])