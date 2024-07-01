import cv2
import numpy as np
import GPIO as GPIO
import threading
import time
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from ui_cheta import Ui_MainWindow

# Параметры ПД-регулятора
Kp = 0.5
Kd = 0.1

# Параметры базовой скорости и сопротивления воды
base_speed = 60

# Предыдущая ошибка для вычисления производнойы
prev_error_x = 0
prev_error_y = 0

# Настройка GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Пины для ШИМ сигнала
left_motor_pin = 17
right_motor_pin = 18

# ЭхоЛокатор
GPIO.setmode(GPIO.BCM)
TRIG = 23
ECHO = 24
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

# Частота ШИМ сигнала
pwm_freq = 1000  # 1 kHz

# Настройка пинов как выходы
GPIO.setup(left_motor_pin, GPIO.OUT)
GPIO.setup(right_motor_pin, GPIO.OUT)

# Создание объектов ШИМ
left_motor_pwm = GPIO.PWM(left_motor_pin, pwm_freq)
right_motor_pwm = GPIO.PWM(right_motor_pin, pwm_freq)

# Запуск ШИМ с 0% скважности
left_motor_pwm.start(0)
right_motor_pwm.start(0)

# Флаги для управления потоками
capture_flag = True
process_flag = True
control_flag = True

# Переменные для хранения данных
frame = None
processed_frame = None


def adjust_brightness_contrast(image, brightness=0, contrast=0):
    beta = brightness
    alpha = contrast / 127 + 1
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted


def find_contours_and_centers(image, lower_bound, upper_bound, min_area=1000):
    mask = cv2.inRange(image, lower_bound, upper_bound)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
    centers = []

    for contour in large_contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append((cx, cy))
        else:
            centers.append(None)
    return large_contours, centers


def get_center_between_two_contours(center1, center2):
    if center1 and center2:
        center_between = ((center1[0] + center2[0]) // 2, (center1[1] + center2[1]) // 2)
        return center_between
    return None


def move_to_point(center_between):
    global prev_error_x, prev_error_y

    # Текущая позиция аппарата (предполагается, что она известна)
    current_x, current_y = 320, 240  # например, центр изображения

    # Вычисление ошибки
    error_x = center_between[0] - current_x
    error_y = center_between[1] - current_y

    # Вычисление производной ошибки
    d_error_x = error_x - prev_error_x
    d_error_y = error_y - prev_error_y

    # ПД-регулятор
    U_x = Kp * error_x + Kd * d_error_x
    U_y = Kp * error_y + Kd * d_error_y

    # Обновление предыдущей ошибки
    prev_error_x = error_x
    prev_error_y = error_y

    # Управление моторами
    motor_left = base_speed + U_x
    motor_right = base_speed - U_x

    # Ограничение значений моторов в пределах 0-100%
    motor_left = max(0, min(100, motor_left))
    motor_right = max(0, min(100, motor_right))

    print(f"Moving to point: {center_between}")
    print(f"Left Motor Speed: {motor_left}%, Right Motor Speed: {motor_right}%")

    left_motor_pwm.ChangeDutyCycle(motor_left)
    right_motor_pwm.ChangeDutyCycle(motor_right)


# Останавливаем оба мотора
def stop_motors():
    left_motor_pwm.ChangeDutyCycle(0)
    right_motor_pwm.ChangeDutyCycle(0)

def move_forward(sex): #Движение вперед в течении определенного времени

    timeout = time.time() + sex  # таймер на sex секунд
    while True:
        left_motor_pwm.ChangeDutyCycle(50)
        right_motor_pwm.ChangeDutyCycle(50)
        if time.time() > timeout:
            break
    stop_motors()

def process_frame(frame):
    # Фильтрация шума
    frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # Коррекция яркости и контрастности
    frame = adjust_brightness_contrast(frame, brightness=30, contrast=30)

    # Преобразование в цветовое пространство HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Определение цветовых диапазонов
    green_lower = np.array([40, 40, 40])
    green_upper = np.array([91, 255, 255])

    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])

    blue_lower = np.array([100, 150, 0])
    blue_upper = np.array([140, 255, 255])

    # Обнаружение контуров и центров для каждого цвета
    green_contours, green_centers = find_contours_and_centers(hsv, green_lower, green_upper)
    yellow_contours, yellow_centers = find_contours_and_centers(hsv, yellow_lower, yellow_upper)
    blue_contours, blue_centers = find_contours_and_centers(hsv, blue_lower, blue_upper)

    # Обработка контуров и центров
    for contours, centers, color in zip(
            [green_contours, yellow_contours, blue_contours],
            [green_centers, yellow_centers, blue_centers],
            [(0, 255, 0), (0, 255, 255), (255, 0, 0)]
    ):
        for contour, center in zip(contours, centers):
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            if center:
                cv2.circle(frame, center, 5, color, -1)
                cv2.putText(frame, f"Center: {center}", (center[0] + 10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            color, 2)
                cv2.putText(frame, f"Top-left: ({x}, {y})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Проверка наличия желтых буйков
    if len(yellow_centers) >= 2:
        largest_yellow_contours = sorted(yellow_contours, key=cv2.contourArea, reverse=True)[:2]
        largest_yellow_centers = [cv2.moments(c) for c in largest_yellow_contours]
        largest_yellow_centers = [(int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])) for m in largest_yellow_centers
                                  if m["m00"] != 0]
        if len(largest_yellow_centers) == 2:
            center_between = get_center_between_two_contours(largest_yellow_centers[0], largest_yellow_centers[1])
            if center_between:
                cv2.circle(frame, center_between, 5, (0, 0, 255),
                           -1)  # Красный цвет для обозначения центра между контурами
                move_to_point(center_between)

    else:

        print("Moving forward")
        # включаем оба двигателя
        left_motor_pwm.ChangeDutyCycle(0)
        right_motor_pwm.ChangeDutyCycle(0)

    # Проверка наличия синего буйка
    if blue_centers:
        blue_center = blue_centers[0]

        timeout = time.time() + 2  # таймер на 2 секунды
        while True:
            move_to_point(blue_center)
            if time.time() > timeout:
                break

        move_square()

        # Используем датчик для определения расстояния до синего буйка
        # distance_to_buoy = get_distance_to_buoy(blue_center)


        # Если дистанция до буйка меньше заданного порога, выполнить вращение вокруг него
        # if distance_to_buoy < 30:  # например, 30 см
        #     rotate_around_buoy(blue_center)
    return frame


def get_distance_to_buoy(center): # Надо добавить локатор
    # Убедитесь, что триггерный пин низкий
    GPIO.output(TRIG, False)

    # Отправка ультразвукового импульса
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    # Измерение времени приема импульса
    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()

    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()

    # Вычисление длительности импульса
    pulse_duration = pulse_end - pulse_start

    # Вычисление расстояния
    distance = pulse_duration * 17150  # Расстояние в сантиметрах
    distance = round(distance, 2)

    return distance


def rotate_around_buoy(center):
    print("Rotating around buoy")
    # Вращаемся, пока синий буй не будет находиться в правой части камеры
    while not is_buoy_on_right(center):
        # Вращение выполняется в течение определенного времени или пока не достигнуто нужное положение
        left_motor_pwm.ChangeDutyCycle(20)
        right_motor_pwm.ChangeDutyCycle(10)

        # обновляем изображение и центр желтого буйка
        ret, frame = cap.read()
        if not ret:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        blue_lower = np.array([100, 150, 0])
        blue_upper = np.array([140, 255, 255])
        _, blue_centers = find_contours_and_centers(hsv, blue_lower, blue_upper)
        if blue_centers:
            center = blue_centers[0]
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def turn_right_90():
    print("90 degres turn")
    left_motor_pwm.ChangeDutyCycle(0)
    right_motor_pwm.ChangeDutyCycle(50)

    time.sleep(1)

    right_motor_pwm.ChangeDutyCycle(0)




def move_square():
    print("Square")
    turn_right_90()
    move_forward(1)
    turn_right_90()
    move_forward(2)
    turn_right_90()
    move_forward(2)
    turn_right_90()
    move_forward(2)
    turn_right_90()
    move_forward(1)
    turn_right_90()

def move_in_circle(radius):  # Или этот или 1 вариант с квадратом
    wheel_distance = 0.1  # Расстояние между колесами (метры)
    left_speed = base_speed * (radius - wheel_distance) / radius
    right_speed = base_speed * (radius + wheel_distance) / radius

    left_motor_pwm(left_speed)
    right_motor_pwm(right_speed)

    time.sleep(10)

    stop_motors()

def is_buoy_on_right(center):
    # Проверяем, находится ли центр желтого буя справа от центра изображения
    image_center_x = frame.shape[1] // 2
    return center[0] > image_center_x


cap = cv2.VideoCapture(0)


def capture_images():
    global frame

    while capture_flag:
        ret, frame = cap.read()
        if not ret:
            break
        frame = process_frame(frame)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.03)  # Маленькая пауза для снижения нагрузки
    cap.release()
    cv2.destroyAllWindows()


def process_images():
    global frame, processed_frame, capture_flag, process_flag, control_flag
    while process_flag:
        if frame is not None:
            processed_frame = process_frame(frame)
        time.sleep(0.03)  # Маленькая пауза для снижения нагрузки


# def control_motors():
#     while control_flag:
#         if processed_frame is not None:
#             move_to_point((320, 240))  # Пример управления, здесь можно реализовать более сложную логику
#         time.sleep(0.03)  # Маленькая пауза для снижения нагрузки

# def other_operations():
#     while True:
#         # Пример других операций
#         print("Performing other operations")
#         time.sleep(1)  # Пауза для имитации выполнения других операций

# Создание и запуск потоков
capture_thread = threading.Thread(target=capture_images)  # image
process_thread = threading.Thread(target=process_images)
# control_thread = threading.Thread(target=control_motors)
# other_thread = threading.Thread(target=other_operations)

capture_thread.start()
process_thread.start()
# control_thread.start()
# other_thread.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    # global capture_flag, process_flag, control_flag
    capture_flag = False
    process_flag = False
    control_flag = False
    capture_thread.join()
    process_thread.join()
    # control_thread.join()
    # other_thread.join()
    stop_motors()
    GPIO.cleanup()
    print("Программа завершена")

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow): #Запускает главное окно приложения.
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.StartBut.clicked.connect(self.on_StartBut_clicked)
        self.StopBut.clicked.connect(self.on_StopBut_clicked)

        # Настройка таймера для обновления изображения с камеры
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.cap = cv2.VideoCapture(0)  # Использование первой камеры
        self.timer.start(20)

    def on_StartBut_clicked(self):
        print("Start")

    def on_StopBut_clicked(self):
        print("Stop")

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Преобразование изображения из BGR (по умолчанию в OpenCV) в RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            step = channel * width
            qImg = QtGui.QImage(frame.data, width, height, step, QtGui.QImage.Format_RGB888)
            self.label.setPixmap(QtGui.QPixmap.fromImage(qImg))

    def closeEvent(self, event):
        # Освобождение ресурсов при закрытии приложения
        self.cap.release()
        event.accept()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())