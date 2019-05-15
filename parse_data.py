from PIL import Image
import piexif
import os, os.path
import numpy as np
import exifread
import torch


exifs = []
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
            continue
        imgDir = os.listdir(os.path.join(path,f))
        img_count += len(imgDir)
        j = imgDir[0]
        ext = os.path.splitext(j)[1]
        # if ext.lower() not in valid_images:
        #     print("not valid")
        #     continue

        fl = open(os.path.join(path,f,j), 'rb')
        tags = exifread.process_file(fl)
        exifs.append(tags)
        print("--------------------- appended tags")
        for tag in tags.keys():
            if tag not in ('JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote'):
                attribute = tag
                if attribute not in attribute_dict:
                    attribute_dict[attribute] = 0

                attribute_dict[attribute] += 1
        
        if img_count >= 500:
            break

    index = 0
    num_atts = 0
    for key, val in attribute_dict.items():
        if val >= 1:
            attribute2index[key] = index
            num_atts += 1
            index += 1


    print("Image Count is " + str(img_count))
    print("Attribute Count is " + str(num_atts))
    print(len(exifs))
    return img_count, num_atts

def attr2ind(attribute):
    return attribute2index[attribute]

# def ind2attr(index):
#     return index2attribute[index]

def exif_vec(im1, im2):
    tags1 = exifs[int(im1)]
    tags2 = exifs[int(im2)]
    vec = np.zeros(len(attribute2index.keys()))
    for tag in tags1:
        if tag not in attribute2index:
            continue
        if tag not in ('JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote'):
            if tag in tags2 and str(tags1[tag]) == str(tags2[tag]):

                vec[attribute2index[tag]] = 1
            else:
                vec[attribute2index[tag]] = 0
    return torch.from_numpy(vec)

