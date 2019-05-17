import numpy as np

def get_data(data_index):
    index = np.load("/Users/matianyi/Desktop/homework3/index.npy")
    index = index.tolist()
    result = {
        "data":[],
        "target":[]
    }
    data_file = open(data_index,encoding="utf-8")
    news = data_file.readline()
    news = news.strip("\n")
    news = news.strip(" ")
    while news != "":
        news = news.replace("\t", " ")
        news = news.replace("， ", "")
        news = news.replace(", ", "")
        news = news.replace("。 ", "")
        news = news.split(" ")
        news_mapping  = []
        for words in news[10:]:
            try:
                news_mapping.append(index[words])
                #print(index[words])
            except:
                pass
        result["data"].append(news_mapping)

        target = news[2:10]
        total = news[1]
        total = total.split(":")
        total = float(total[1])
        target_mapping = []
        for data in target:
            data = data.split(":")
            data = float(data[1])
            target_mapping.append(data/total)
        max_m = -10000
        sss = -1
        for ss in range(0, len(target_mapping)):
            if target_mapping[ss] > max_m:
                max_m = target_mapping[ss]
                sss = ss
            target_mapping[ss] = 0
        target_mapping[sss] = 1
        #print(target_mapping)

        result["target"].append(sss)
        print(sss)
        news = data_file.readline()
        news = news.strip("\n")
        news = news.strip(" ")
    return result

resul = get_data("/Users/matianyi/Desktop/homework3/sinanews.test")
print(len(resul["data"]))
print(len(resul["target"]))
#filez = open("/Users/matianyi/Desktop/gg.txt","w",encoding="utf-8")
#filez.write(str(resul))
resul = np.array(resul)
np.save("/Users/matianyi/Desktop/homework3/test_set.npy",resul)

