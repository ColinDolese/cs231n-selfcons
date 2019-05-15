from PIL import Image
import piexif
import os, os.path
import numpy as np

#exifs = []
path = "./FlickrCentral"
valid_images = [".jpg",".gif",".png",".tga"]

attribute2index = {}
index2attribute = {}

def getAttributes():
    attribute_dict = {}

    img_count = 0
    for f in os.listdir(path):
        print("Loading image " + str(img_count))
        if img_count == 14344:
            img_count -= 1
            continue
        if f == ".DS_Store":
            continue
        imgDir = os.listdir(os.path.join(path,f))
        img_count += len(imgDir)
        j = imgDir[0]
        ext = os.path.splitext(j)[1]
        if ext.lower() not in valid_images:
            continue
        im = Image.open(os.path.join(path,f,j))
        if "exif" not in im.info:
            continue
        exif_dict = piexif.load(im.info["exif"])
        #for ifd in ("0th", "Exif", "GPS", "1st"):
        for ifd in (["Exif"]):
            for tag in exif_dict[ifd]:
                #attribute = ifd + "_" + str(tag) + "_" + piexif.TAGS[ifd][tag]["name"] 
                attribute = piexif.TAGS[ifd][tag]["name"] 
                value = exif_dict[ifd][tag]
                #print(piexif.TAGS[ifd][tag]["name"], exif_dict[ifd][tag])

                if attribute not in attribute_dict:
                    attribute_dict[attribute] = 0

                attribute_dict[attribute] += 1
        #exifs.append(exif_dict)
        

    index = 0
    for key, val in attribute_dict.items():
        index2attribute[index] = key
        attribute2index[key] = index
        index += 1

    return img_count, len(attribute_dict.keys())

# def attr2ind(attribute):
#     return attribute2index[attribute]

# def ind2attr(index):
#     return index2attribute[index]

# def get_attribute_vec(imgIndex):
#     exif_dict = exifs[imgIndex]
#     vec = np.zeros(len(attribute2index.keys()))
#     for tag in exif_dict["Exif"]:
#         attribute = piexif.TAGS["Exif"][tag]["name"]
#         vec[attr2ind(attribute)] = 1

#     return vec

# getAttributes()

