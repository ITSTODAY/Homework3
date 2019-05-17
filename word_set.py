index1 = "sinanews.train"
index2 = "vocab"
def get_set(index_in, index_out):
    
    f = open(index_in)
    
    a = f.readline()
    a = a.strip("\n")
    a = a.strip(" ")
    vocab = set([])
    while(a!=""):
        a = a.replace("\t"," ")
        a = a.replace(", ","")
        a = a.replace("。 ","")
        a = a.replace("， ","")
        a = a.split(" ")
        b = a[10:]
        vocab = vocab | set(b)
        a = f.readline()
        a = a.strip("\n")
        a = a.strip(" ")
    vocab_s = ""
    for word in vocab:
        vocab_s = vocab_s+word
        vocab_s = vocab_s+" "
    vocab_s = vocab_s.strip(" ")
    f.close()
    f = open(index_out,"w")
    f.write(vocab_s)

get_set(index1,index2)