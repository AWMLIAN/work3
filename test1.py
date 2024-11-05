# -*- coding: utf-8 -*-


import cv2
import numpy as np
import os
import shutil


# 采集自己的人脸数据
def generator(data):
    '''
    打开摄像头，读取帧，检测该帧图像中的人脸，并进行剪切、缩放
    生成图片满足以下格式：
    1.灰度图，后缀为 .png
    2.图像大小相同
    params:
        data:指定生成的人脸数据的保存路径
    '''

    name = input('my name:')
    # 如果路径存在则删除路径
    path = os.path.join(data, name)
    if os.path.isdir(path):
        shutil.rmtree(path)
    # 创建文件夹
    os.mkdir(path)
    # 创建一个级联分类器
    face_casecade = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')
    # 打开摄像头
    camera = cv2.VideoCapture(0)
    cv2.namedWindow('Dynamic')
    # 计数
    count = 1

    while (True):
        # 读取一帧图像
        ret, frame = camera.read()
        if ret:
            # 转换为灰度图
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 人脸检测
            face = face_casecade.detectMultiScale(gray_img, 1.3, 5)
            for (x, y, w, h) in face:
                # 在原图上绘制矩形
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                # 调整图像大小
                new_frame = cv2.resize(frame[y:y + h, x:x + w], (92, 112))
                # 保存人脸
                cv2.imwrite('%s/%s.png' % (path, str(count)), new_frame)
                count += 1
            cv2.imshow('Dynamic', frame)
            # 按下q键退出
            if cv2.waitKey(100) & 0xff == ord('q'):
                break
    camera.release()
    cv2.destroyAllWindows()


# 载入图像   读取ORL人脸数据库，准备训练数据
def LoadImages(data):
    '''
    加载图片数据用于训练
    params:
        data:训练数据所在的目录，要求图片尺寸一样
    ret:
        images:[m,height,width]  m为样本数，height为高，width为宽
        names：名字的集合
        labels：标签
    '''
    images = []
    names = []
    labels = []

    label = 0

    # 遍历所有文件夹
    for subdir in os.listdir(data):
        subpath = os.path.join(data, subdir)
        # print('path',subpath)
        # 判断文件夹是否存在
        if os.path.isdir(subpath):
            # 在每一个文件夹中存放着一个人的许多照片
            names.append(subdir)
            # 遍历文件夹中的图片文件
            for filename in os.listdir(subpath):
                imgpath = os.path.join(subpath, filename)
                img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # cv2.imshow('1',img)
                # cv2.waitKey(0)
                images.append(gray_img)
                labels.append(label)
            label += 1
    images = np.asarray(images)
    # names=np.asarray(names)
    labels = np.asarray(labels)
    return images, labels, names


# 检验训练结果
def FaceRec(data):
    # 加载训练的数据
    X, y, names = LoadImages(data)
    # print('x',X)
    model = cv2.face.EigenFaceRecognizer_create()
    model.train(X, y)

    # 打开摄像头
    camera = cv2.VideoCapture(0)
    cv2.namedWindow('Dynamic')

    # 创建级联分类器
    face_casecade = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')

    while (True):
        # 读取一帧图像
        # ret:图像是否读取成功
        # frame：该帧图像
        ret, frame = camera.read()
        # 判断图像是否读取成功
        # print('ret',ret)
        if ret:
            # 转换为灰度图
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 利用级联分类器鉴别人脸
            faces = face_casecade.detectMultiScale(gray_img, 1.3, 5)

            # 遍历每一帧图像，画出矩形
            for (x, y, w, h) in faces:
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 蓝色
                roi_gray = gray_img[y:y + h, x:x + w]

                try:
                    # 将图像转换为宽92 高112的图像
                    # resize（原图像，目标大小，（插值方法）interpolation=，）
                    roi_gray = cv2.resize(roi_gray, (92, 112), interpolation=cv2.INTER_LINEAR)
                    params = model.predict(roi_gray)
                    print('Label:%s,confidence:%.2f' % (params[0], params[1]))
                    '''
                    putText:给照片添加文字
                    putText(输入图像，'所需添加的文字'，左上角的坐标，字体，字体大小，颜色，字体粗细)
                    '''
                    cv2.putText(frame, names[params[0]], (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
                except:
                    continue

            cv2.imshow('Dynamic', frame)

            # 按下q键退出
            if cv2.waitKey(100) & 0xff == ord('q'):
                break
    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    data = "D:/frac_renlian"
    generator(data)
    FaceRec(data)

