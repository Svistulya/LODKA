import cv2
import numpy as np
# import RPi.GPIO as GPIO

# Параметры ПД-регулятора
Kp = 0.5
Kd = 0.1

# Параметры базовой скорости и сопротивления воды
base_speed = 50

# Предыдущая ошибка для вычисления производной
prev_error_x = 0
prev_error_y = 0


# Настройка GPIO
# GPIO.setmode(GPIO.BCM)
# GPIO.setwarnings(False)

# # Пины для ШИМ сигнала
# left_motor_pin = 17
# right_motor_pin = 18

# # Частота ШИМ сигнала
# pwm_freq = 1000  # 1 kHz

# # Настройка пинов как выходы
# GPIO.setup(left_motor_pin, GPIO.OUT)
# GPIO.setup(right_motor_pin, GPIO.OUT)

# # Создание объектов ШИМ
# left_motor_pwm = GPIO.PWM(left_motor_pin, pwm_freq)
# right_motor_pwm = GPIO.PWM(right_motor_pin, pwm_freq)

# # Запуск ШИМ с 0% скважности
# left_motor_pwm.start(0)
# right_motor_pwm.start(0)

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

    # left_motor_pwm.ChangeDutyCycle(motor_left)
    # right_motor_pwm.ChangeDutyCycle(motor_right)

    # Останавливаем оба мотора
# def stop_motors():
#     left_motor_pwm.ChangeDutyCycle(0)
#     right_motor_pwm.ChangeDutyCycle(0)

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
                cv2.putText(frame, f"Center: {center}", (center[0] + 10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, f"Top-left: ({x}, {y})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Проверка наличия желтого буйков
    if len(yellow_centers) >= 2:
        largest_yellow_contours = sorted(yellow_contours, key=cv2.contourArea, reverse=True)[:2]
        largest_yellow_centers = [cv2.moments(c) for c in largest_yellow_contours]
        largest_yellow_centers = [(int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])) for m in largest_yellow_centers if m["m00"] != 0]
        if len(largest_yellow_centers) == 2:
            center_between = get_center_between_two_contours(largest_yellow_centers[0], largest_yellow_centers[1])
            if center_between:
                cv2.circle(frame, center_between, 5, (0, 0, 255), -1)  # Красный цвет для обозначения центра между контурами
                move_to_point(center_between)
    # elif len(yellow_centers) == 1:
    #     move_to_point(yellow_centers[0])
    else:
        
        
        # center_between = get_center_between_two_contours(yellow_centers[0], yellow_centers[1])
        # if center_between:
        #     cv2.circle(frame, center_between, 5, (0, 0, 255), -1)  # Красный цвет для обозначения центра между контурами
        #     move_to_point(center_between)
    
    # else:
        # Если нет двух желтых буйков, двигаться вперед
        print("Moving forward")
        # включаем оба двигателя
        # left_motor_pwm.ChangeDutyCycle(base_speed)
        # right_motor_pwm.ChangeDutyCycle(base_speed)
        motor_left = base_speed
        motor_right = base_speed

    # Проверка наличия синего буйка
    if blue_centers:
        blue_center = blue_centers[0]
        move_to_point(blue_center)

        # Используем датчик для определения расстояния до синего буйка
        distance_to_buoy = get_distance_to_buoy(blue_center)

        # Если дистанция до буйка меньше заданного порога, выполнить вращение вокруг него
        if distance_to_buoy < 30:  # например, 30 см
            rotate_around_buoy(blue_center)
    return frame

def get_distance_to_buoy(center):
    # Здесь нужно использовать данные с датчика для определения расстояния до буйка
    # В данном примере функция возвращает фиктивное значение
    return 25  # например, 25 см

def rotate_around_buoy(center):
    print("Rotating around buoy")
    # Вращаемся, пока синий буй не будет находиться в правой части камеры
    while not is_buoy_on_right(center):
        # Вращение выполняется в течение определенного времени или пока не достигнуто нужное положение
        # left_motor_pwm.ChangeDutyCycle(base_speed)
        # right_motor_pwm.ChangeDutyCycle(0)
        motor_left = base_speed
        motor_right = -base_speed
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

def is_buoy_on_right(center):
    # Проверяем, находится ли центр желтого буя справа от центра изображения
    image_center_x = frame.shape[1] // 2
    return center[0] > image_center_x

# Захват видео с камеры
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = process_frame(frame)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # if is_buoy_on_right(center):
    #         left_motor_pwm.ChangeDutyCycle(40)
    #         right_motor_pwm.ChangeDutyCycle(60)
    # else:
    #     break

cap.release()
cv2.destroyAllWindows()