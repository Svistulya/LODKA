import cv2
import numpy as np

# Параметры ПД-регулятора
Kp = 0.5
Kd = 0.1

# Параметры базовой скорости и сопротивления воды
base_speed = 50
R = 0.05

# Предыдущая ошибка для вычисления производной
prev_error_x = 0
prev_error_y = 0

def adjust_brightness_contrast(image, brightness=0, contrast=0):
    beta = brightness
    alpha = contrast / 127 + 1
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

def find_contours_and_centers(image, lower_bound, upper_bound, min_area=1500):
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

def process_frame(frame):
    # Фильтрация шума
    frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # Коррекция яркости и контрастности
    frame = adjust_brightness_contrast(frame, brightness=30, contrast=30)

    # Преобразование в цветовое пространство HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Определение цветовых диапазонов
    green_lower = np.array([40, 40, 40])
    green_upper = np.array([80, 255, 255])

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
                # cv2.putText(frame, f"Center: {center}", (center[0] + 10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, f"Top-left: ({x}, {y})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Проверка наличия синих буйков
    if len(blue_centers) >= 2:
        center_between = get_center_between_two_contours(blue_centers[0], blue_centers[1])
        if center_between:
            cv2.circle(frame, center_between, 5, (0, 0, 255), -1)  # Красный цвет для обозначения центра между контурами
            move_to_point(center_between)
    else:
        # Если нет двух синих буйков, двигаться вперед
        print("Moving forward")
        # включаем оба двигателя
        motor_left = base_speed
        motor_right = base_speed

    # Проверка наличия желтого буйка
    if yellow_centers:
        yellow_center = yellow_centers[0]
        move_to_point(yellow_center)

        # Используем датчик для определения расстояния до желтого буйка
        distance_to_buoy = get_distance_to_buoy(yellow_center)

        # Если дистанция до буйка меньше заданного порога, выполнить вращение вокруг него
        if distance_to_buoy < 30:  # например, 30 см
            rotate_around_buoy(yellow_center)
    return frame

def get_distance_to_buoy(center):
    # Здесь нужно использовать данные с датчика для определения расстояния до буйка
    # В данном примере функция возвращает фиктивное значение
    return 25  # например, 25 см

def rotate_around_buoy(center):
    print("Rotating around buoy")
    # Здесь необходимо реализовать алгоритм вращения вокруг буйка
    # Например, можно использовать разное направление вращения моторов
    motor_left = base_speed
    motor_right = -base_speed
    # Вращение выполняется в течение определенного времени или пока не достигнуто нужное положение

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = process_frame(frame)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()