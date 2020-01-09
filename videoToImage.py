import os
from cv2 import cv2 as cv


def createImage(videofile, fps=10):

    names = videofile.split(".", 1)[0]
    filename = createSortFile(names)
    filename = filename + names
    # 打卡视频
    video = cv.VideoCapture(videofile)

    count = 0
    name = 1
    rval = True
    while rval:
        rval, img = video.read()
        if (count % fps == 0 and rval):
            print(rval, "正在保存图片:张数 = ", name)
            cv.imwrite(filename + str(name) + ".jpg", img)
            name = name + 1
            pass
        count = count + 1
        cv.waitKey(1)
        pass

    video.release


def createSortFile(filename):
    # 创建分类文件夹
    filename = "images/" + filename
    isExists = os.path.exists(filename)
    if not isExists:
        os.makedirs(filename)
    return filename + "/"


createImage("21.mp4")
createImage("22.mp4")
createImage("11.mp4")
createImage("12.mp4")
