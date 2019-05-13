import csv
import requests
import os
import sys
import time
import random 
def put_images(FILE_NAME):
    urls=[]
    with open(FILE_NAME,newline="") as csvfile:
        doc=csv.reader(csvfile,delimiter=",")
        for row in doc:
            if row[1].startswith("https"):
                urls.append(row[1])
    if not os.path.isdir(os.path.join(os.getcwd(),FILE_NAME.split("_")[0])):
        os.mkdir(FILE_NAME.split("_")[0])
    t0=time.time()

    imgNum = 0
    for url in enumerate(urls):

        numToAdd = random.randint(3, 10)
        rootPath = os.path.join(os.getcwd(),FILE_NAME.split("_")[0])
        rootPath += "/" + str(imgNum)
        print(rootPath)
        if not os.path.isdir(rootPath):
            os.mkdir(FILE_NAME.split("_")[0] + "/" + str(imgNum))
        for i in range(numToAdd):
            print("Starting download {} of ".format(url[0]+1),len(urls))
            try:
                resp=requests.get(url[1],stream=True)
                path_to_write=os.path.join(os.getcwd(),FILE_NAME.split("_")[0],str(imgNum) + "/", url[1].split("/")[-1])
                strList = path_to_write.split(".jpg")
                path_to_write = strList[0] + "_" + str(i) + ".jpg"
                print(path_to_write)
                outfile=open(path_to_write,'wb')
                outfile.write(resp.content)
                outfile.close()
                print("Done downloading {} of {}".format(url[0]+1,len(urls)))
            except:
                print("Failed to download url number {}".format(url[0]))

        imgNum += 1
        
        if imgNum == 5:
            break

    t1=time.time()
    print("Done with download, job took {} seconds".format(t1-t0))
def main():
    FILE_NAME=sys.argv[1]
    put_images(FILE_NAME)
if __name__=='__main__':
    main()



