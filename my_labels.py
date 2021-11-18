import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import shutil

# 源代码sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
sets = [('data', 'train'), ('data', 'test'), ('data', 'val')]  # 改成自己建立的myData

classes = ["aircraft", "oiltank"]  # 改成自己的类别


def convert(size, box):  # 640*480 1 192 613 480 xmin ymin xmax ymax
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(year, image_id, mode):
    in_file = open('data/Annotations/%s.xml' % (image_id))  # 源代码VOCdevkit/VOC%s/Annotations/%s.xml
    out_file = open('data/labels/%s/%s.txt' % (mode, image_id), 'w')  # 源代码VOCdevkit/VOC%s/labels/%s.txt
    tree = ET.parse(in_file)  # 载入数据
    root = tree.getroot()  # 获取根节点
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):  # 用for遍历一张图片的每个标签，而图片只有一个weight height，故不用遍历
        #difficult = obj.find('difficult').text
        cls = obj.find('name').text
       # if cls not in classes or int(difficult) == 1:
         #   continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)  # (x, y, w, h) float
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')  # 用空格连接 cls_id x y w h\n


wd = getcwd()  # 获取当前目录
path1 = '%s/data/images/' % wd
path2 = '%s/data/ImageSets/' % wd

for year, image_set in sets:  # 读取sets = [('myData', 'train'), ('myData', 'test')]
    if not os.path.exists('data/labels/'):  # 改成自己建立的myData
        os.makedirs('data/labels/')
    image_ids = open('data/ImageSets/Main/%s.txt' % (image_set)).read().strip().split()  # train.txt split默认以任何形式分隔符分割，这里是\n
    # print(image_ids)
    list_file = open('data/%s_%s.txt' % (year, image_set), 'w')  # 使用w模式打开文件 如果文件存在 会覆盖并打开；如果文件不存在 会创建一个文件 然后打开

    if image_set == 'train':
        for image_id in image_ids:
            list_file.write('%s/data/images/train/%s.jpg\n' % (wd, image_id))  # 往myData_train中写入
            convert_annotation(year, image_id, 'train')
            shutil.copy(path2+image_id+'.jpg', path1+'train/'+image_id+'.jpg')
    elif image_set == 'test':
        for image_id in image_ids:
            list_file.write('%s/data/images/test/%s.jpg\n' % (wd, image_id))  # 往myData_train中写入
            convert_annotation(year, image_id, 'test')
            shutil.copy(path2 + image_id + '.jpg', path1 + 'test/' + image_id + '.jpg')
    elif image_set == 'val':
        for image_id in image_ids:
            list_file.write('%s/data/images/val/%s.jpg\n' % (wd, image_id))  # 往myData_train中写入
            convert_annotation(year, image_id, 'val')
            shutil.copy(path2 + image_id + '.jpg', path1 + 'val/' + image_id + '.jpg')
    list_file.close()
