import numpy as np
from concurrent.futures import ThreadPoolExecutor
import cv2
import torch
from ultralytics import YOLO
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from screeninfo import get_monitors
import robots
import scene


def create_point(*params) -> None:
    global goal_point
    global handle_mode
    if params[0] == cv2.EVENT_LBUTTONDBLCLK:
        goal_point = params[1] * 2, params[2] * 2
        return None
    elif params[0] == cv2.EVENT_RBUTTONDBLCLK:
        handle_mode = False if handle_mode else True
    else:
        return None


def coppeliasim_remoteAPI() -> None:
    global frame
    global leftside_speed
    global rightside_speed
    global run_flag
    global roi
    # Инициализируем клиент CoppeliaSim, загружаем сцену
    copsim_scene = scene.coppeliasim_remoteAPI(scene_name='/base_2cam.ttt')
    while run_flag:
        frame = copsim_scene.get_frame()
        cv2_frame = copsim_scene.transform_cv2_frame(frame, resize_ratio=2)
        # Рисуем целеуказатель
        if goal_point:
            cv2.circle(cv2_frame, (goal_point[0] // 2, goal_point[1] // 2), 2, (0, 0, 255), 2)
        if handle_mode:
            cv2.putText(cv2_frame, 'Handle mode ON', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (0, 0, 255), 2, cv2.LINE_AA)
            leftside_speed, rightside_speed = copsim_scene.handle_mode_movement()
        copsim_scene.update_motors(leftside_speed, rightside_speed)
        cv2.imshow('out', cv2_frame)
        cv2.setMouseCallback('out', create_point)
        if roi is not None:
            cv2.imshow('roi', roi)
        if cv2.waitKey(10) & 0xFF == 27: 
            copsim_scene.stop()
            run_flag = False


def robot_movement_proc() -> None:
    '''
    Отдельный процесс, отвечающий за управление конкретным роботом. В объекте,
    получаемом от обработчика потокового видео, ищет координаты робота, после
    чего рассчитывает траекторию его движения и передает управляющие команды.
    '''
    global leftside_speed
    global rightside_speed
    global frame
    global run_flag
    global goal_point
    global handle_mode
    global roi
    # Создаем экземпляр класса "робот", классы введены для поддержки нескольких
    # роботов на одной сцене, для каждого отдельного робота создается экземпляр
    robot_1 = robots.Robots(speed=4, model='./Models/pose.pt')
    # Переменная для счетчика кадров, в которых робот не обнаружен
    missed_frames_cnt = 0
    # Основной цикл управления роботом
    while run_flag:
        if handle_mode:
            goal_point = None
            continue
        # Если робота "не видно" 10+ кадров, вращаем его до сизого дыма
        if missed_frames_cnt > 10:
            #robot_1.robot_move_backward()
            leftside_speed, rightside_speed = 1, 1
        # Робот ищет свои координаты в результатах обработки кадра моделью
        roi = robot_1.find_itself(frame=frame)
        if roi is not None:
            missed_frames_cnt = 0
        else:
            missed_frames_cnt += 1
            continue
        # Передаем роботу координаты целевой точки, строим маршрут
        # Если робот успешно построил маршрут, он двигается по нему
        if robot_1.check_way(goal_point=goal_point):
            leftside_speed, rightside_speed = robot_1.robot_action()
        # Если робот достиг целевой точки или маршрут невозможен,
        # то останавливаем робота и сбрасываем целевую точку
        else:
            goal_point = None
            leftside_speed, rightside_speed = (0, 0)
    # Выводим последние 400 записей истории
    print(robot_1.show_history())


run_flag = True         # флаг для остановки программы
frame = None            # переменная для передачи текущего кадра с виртуальной камеры
goal_point = None       # Координаты целевой точки
leftside_speed = 0      # скорость колес левого борта
rightside_speed = 0     # скорость колес правого борта
handle_mode = False
roi = None


# Запуск модулей в параллельных процессах
with ThreadPoolExecutor(max_workers=4) as executor:
    coppeliasim_task = executor.submit(coppeliasim_remoteAPI)
    robot_processing = executor.submit(robot_movement_proc)
    # Результаты работы модулей после остановки
    print(coppeliasim_task.result())
    print(robot_processing.result())
