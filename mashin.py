# Very often, images are treated as an array of RGB pixels. Although this idea appears
# to be relatively intuitive, it is not optimal for the problem we have to solve.
# In RGB, the color of a pixel is determined by the saturation of red, green and blue.
# Thus, choosing a range of shades of the same color becomes not the easiest task.
# Things are different with the HSV format. This color scheme is determined by three components:
# Hue - color tone;
# Saturation - saturation;
# Value - brightness.
# In the HSV scheme, the base color can be selected using the Hue component (for example, red, orange, etc.).
# The other two components allow you to adjust the saturation and brightness of the base color,
# making it more saturated or dull, lighter or darker.
# These properties of HSV make it easy to define ranges that can capture areas of the desired color and its shades.

import cv2
import numpy as np
from matplotlib import pyplot as plt

# создается объект с именем image куда помещается изображение test2
# используется библиотека CV2 (opencv) метод imread, он загружает изображение
# в объект
# 1. Файл с изображением помещени в папку с проектом
# 2 указать полный путь до изоражения
image = cv2.imread("test2.jpg")
# создание окна для вывода изображния "original" подпись окна
cv2.imshow("original", image)
# Выводи изображение и ожидам нажатия любой клавиши
cv2.waitKey(0) 
cv2.destroyAllWindows()      ##### закрываем окно

#dst = cv2.fastNlMeansDenoisingColored(image , None, 10, 10, 7, 15) 

#cv2.imshow("shum", dst)
# Выводи изображение и ожидам нажатия любой клавиши
#cv2.waitKey(0)


# Создаем объект blurred_image и применяю Гаусов фильтр размытия
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
#  создание окна для вывода изображния "blurred" подпись окна
cv2.imshow("blurred", blurred_image)
# Выводи изображение и ожидам нажатия любой клавиши
cv2.waitKey(0)
cv2.destroyAllWindows()      ##### закрываем окно

# Конвертирует цветовую палитру размытого изображения
# из RGB->HSV (RGB convet HSV)
# библиотеке opencv удобнее работать с цветовой палитной HSV
hsv_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)
#  создание окна для вывода изображния "hsv" подпись окна
cv2.imshow("hsv", hsv_image)
# Выводи изображение и ожидам нажатия любой клавиши   
cv2.waitKey(0)
cv2.destroyAllWindows()      ##### закрываем окно

# Создаем верхнию и нижнию границу цветов
hsv_min = np.array((25, 25, 25), np.uint8)  # Нижняя граница зеленого
hsv_max = np.array((28, 255, 255), np.uint8)  # Верхняя граница зеленого

# Создаем объект "green_mask"  в него сохраняем результат
# применения маски зеленого цвета
green_mask = cv2.inRange(hsv_image, hsv_min, hsv_max)
#  создание окна для вывода изображния "mask" подпись окна
cv2.imshow("mask", green_mask)
# Выводи изображение и ожидам нажатия любой клавиши
cv2.waitKey(0)
cv2.destroyAllWindows()      ##### закрываем окно


thresh = cv2.adaptiveThreshold(green_mask, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imshow('Threshold Image', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

# применяем метод findContours находим все контура согласно применной маски
#                              указываем копию маски; алгорим поиска контуров; алгорим апроксимация конутров
contours, hierarchy = cv2.findContours(green_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
# cv2.imshow('Contours', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#Сортируем конура по по размеру пплощади
sorted_contur = sorted(contours, key=cv2.contourArea, reverse=True)

# определяем кооридина 0 конутра верхнего левого угла контура и его размервы ширену и высоту
print ("Количество контуров")
print (len(sorted_contur))
# contours = contours[0] if len(contours) == 2 else contours[1]
for c in sorted_contur:
    x, y, w, h, = cv2.boundingRect(c)
    print(x, y, w, h)
    # рисую окружность по указанным кооридинатам
    cv2.circle(image,(x,y),10,(255,0,0),-1)
    # # Находим и рисуем точку в центре объета (изображения)
    cv2.circle(image,(int (x+w/2),int (y+h/2)),1, (0,0,255), 3)
# обводим все найденные контуры
cv2.drawContours(image, sorted_contur, -1, (255, 0, 0), 1, cv2.LINE_AA, hierarchy, 1)
# определяем кооридинаты и размеры 1 конутра
# x, y, w, h, = cv2.boundingRect(sorted_contur[1])
# print(x, y, w, h)
# Обвожу 1 конут прямоуголиников
#cv2.rectangle(image, (x,y), (int(x+w), int(y+h)),(0,0,255),6)
#  cv2.drawContours(image, sorted_contur, -1, (255, 0, 0), 1, cv2.LINE_AA, hierarchy, 1)



# # Create a window. Output image. Signing the window
cv2.imshow('contours', image)
# # Wait for the button to be pressed to move to the next command
cv2.waitKey(0)
cv2.destroyAllWindows()