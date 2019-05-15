from PIL import Image
import piexif
import os, os.path
import numpy as np
import exifread


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
        if f == ".DS_Store":
            print("is ds_store")
            continue
        imgDir = os.listdir(os.path.join(path,f))
        img_count += len(imgDir)
        j = imgDir[0]
        ext = os.path.splitext(j)[1]
        if ext.lower() not in valid_images:
            print("not valid")
            continue
        # im = Image.open(os.path.join(path,f,j))
        # if "exif" not in im.info:
        #     continue
        # exif_dict = piexif.load(im.info["exif"])
        # #for ifd in ("0th", "Exif", "GPS", "1st"):
        # for ifd in (["Exif"]):
        #     for tag in exif_dict[ifd]:
        #         #attribute = ifd + "_" + str(tag) + "_" + piexif.TAGS[ifd][tag]["name"] 
        #         attribute = piexif.TAGS[ifd][tag]["name"] 
        #         value = exif_dict[ifd][tag]
        #         #print(piexif.TAGS[ifd][tag]["name"], exif_dict[ifd][tag])

        #         if attribute not in attribute_dict:
        #             attribute_dict[attribute] = 0

        #         attribute_dict[attribute] += 1
        # #exifs.append(exif_dict)
        fl = open(os.path.join(path,f,j), 'rb')
        tags = exifread.process_file(fl)
        for tag in tags.keys():
            if tag not in ('JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote'):
                attribute = tag
                if attribute not in attribute_dict:
                    attribute_dict[attribute] = 0

                attribute_dict[attribute] += 1
        
        if img_count >= 1000:
            break

    index = 0
    num_atts = 0
    for key, val in attribute_dict.items():
        if val >= 200:
            num_atts += 1
        index += 1


    print("Image Count is " + str(img_count))
    print("Attribute Count is " + str(num_atts))
    return img_count, num_atts

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

getAttributes()

