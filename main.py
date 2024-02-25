import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import cv2
import torch
from ultralytics import YOLO
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import robots


def create_point(*params) -> None:
    '''Создаю целевую точку для робота двойным кликом мыши на кадре'''
    global goal_point
    event = params[0]
    x_ = params[1]
    y_ = params[2]
    if (event == cv2.EVENT_LBUTTONDBLCLK): 
        goal_point = x_, y_
        return None
    else:
        return None


def coppeliasim_remoteAPI() -> None:
    '''
    Отдельный процесс, отвечающий за взаимодействие с эмулятором CoppeliaSim.
    Запускается и выполняется параллельно и асинхронно с обработкой видео.
    Управление роботом осуществляется через глобальные переменные:
    - leftside_speed - скорость колес левого борта.
    - rightside_speed - скорость колес правого борта.
    Передача изображения с камеры идет через глобальную переменную frame.
    Это сделано для имитации каналов дистанционного приема/передачи информации. 
    '''
    global frame
    global cam_num
    global leftside_speed
    global rightside_speed
    global run_flag
    # Инициализируем клиент CoppeliaSim, загружаем сцену
    client = RemoteAPIClient()
    sim = client.require('sim')
    sim.loadScene(sim.getStringParam(
        sim.stringparam_scenedefaultdir) + '/base.ttt')
    # Создаем объекты двигателей роботов и виртуальных камер
    frontleft_motor = sim.getObject('./front_left_wheel')
    frontright_motor = sim.getObject('./front_right_wheel')
    backleft_motor = sim.getObject('./back_left_wheel')
    backright_motor = sim.getObject('./back_right_wheel')
    Cam_1_Handle = sim.getObject(f'./building/cam_1')
    Cam_2_Handle = sim.getObject(f'./building/cam_2')
    # Запускаем симуляцию и цикл управления объектами сцены
    sim.startSimulation()
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #out = cv2.VideoWriter('output.avi', fourcc, 15.0, (1280, 960))
    while run_flag:
        if cam_num == 1:
            img, res = sim.getVisionSensorImg(Cam_1_Handle)
        elif cam_num == 2:
            img, res = sim.getVisionSensorImg(Cam_2_Handle)
        img_RGB = np.frombuffer(img, dtype=np.uint8).reshape(res[1], res[0], 3)
        frame = img_RGB[::-1, :, :]
        # Меняем цветовые форматы для корректного отображения
        img_BGR = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Рисуем целеуказатель
        cv2.circle(img_BGR, goal_point, 2, (0, 0, 255), 2)
        cv2.imshow('frame', img_BGR)
        #out.write(img_BGR)
        cv2.setMouseCallback('frame', create_point)
        if cv2.waitKey(1) & 0xFF == 27: 
            run_flag = False
        sim.setJointTargetVelocity(frontleft_motor, leftside_speed)
        sim.setJointTargetVelocity(frontright_motor, rightside_speed)
        sim.setJointTargetVelocity(backleft_motor, leftside_speed)
        sim.setJointTargetVelocity(backright_motor, rightside_speed)
    sim.stopSimulation()
    #out.release()
    cv2.destroyAllWindows()


def detect_objects() -> None:
    '''
    Отдельный процесс, отвечающий за обнаружение координат всех объектов в кадре
    с помощью предобученной модели и трансляцию в виде глобальных переменных.
    Должен работать на GPU.
    '''
    global frame
    global boxes
    global run_flag
    # Автоматический выбор устройства. Крайне рекомендуется иметь CUDA.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Загрузка модели YOLO
    model = YOLO('./Models/detect.pt')
    #robot_1 = robots.Robots()
    # Запуск цикла обработки видеопотока
    while run_flag:
        #start_time = time.time()
        # Обрабатываем кадр, ищем объекты, возвращаем boxes
        results = model.track(frame,
                              device=device,
                              verbose=False,
                              max_det=1,
                              imgsz=640)
        if results:
            boxes = results[0].boxes
        #print(time.time() - start_time)


def collision_watchdog() -> None:
    '''
    Заготовка для отдельного процесса, который препятствует ситуациям столкновения
    отслеживаемых объектов между собой при выполнении операций. Получает результаты
    обработки кадра, выполняется на GPU и блокирует движение робота, если его
    ограничивающая рамка опасно сближается с ограничивающими рамками других объектов.
    '''
    pass


def robot_movement_proc() -> None:
    '''
    Отдельный процесс, отвечающий за управление конкретным роботом. В объекте,
    получаемом от обработчика потокового видео, ищет координаты робота, после
    чего рассчитывает траекторию его движения и передает управляющие команды.
    '''
    global leftside_speed
    global rightside_speed
    global boxes
    global frame
    global cam_num
    global run_flag
    global goal_point
    # Создаем экземпляр класса "робот", классы введены для поддержки нескольких
    # роботов на одной сцене, для каждого отдельного робота создается экземпляр
    robot_1 = robots.Robots(speed=8)
    # Переменная для счетчика кадров, в которых робот не обнаружен
    missed_frames_cnt = 0
    # Основной цикл управления роботом
    while run_flag:
        # Если робота "не видно" 10+ кадров, вращаем его до сизого дыма
        if missed_frames_cnt > 10:
            #robot_1.robot_move_backward()
            leftside_speed, rightside_speed = 1, 1
        # Робот ищет свои координаты в результатах обработки кадра моделью
        if robot_1.find_itself(boxes=boxes):
            missed_frames_cnt = 0
            robot_1.find_keypoints_vec(frame=frame)
        else:
            missed_frames_cnt += 1
            continue
        # Проверяем условия переключения камер и переключаем при необходимости
        cam_checking = robot_1.check_cam(frame.shape, cam_num)
        if cam_checking:
            cam_num = cam_checking
            continue
        # Передаем роботу координаты целевой точки, строим маршрут
        # Если робот успешно построил маршрут, он двигается по нему
        if robot_1.find_way(frame=frame, goal_point=goal_point):
            leftside_speed, rightside_speed = robot_1.robot_action()
        # Если робот достиг целевой точки или маршрут невозможен,
        # то останавливаем робота и сбрасываем целевую точку
        else:
            goal_point = None
            leftside_speed, rightside_speed = (0, 0)
    # Выводим последние 400 записей истории
    robot_1.show_history()


run_flag = True         # флаг для остановки программы
frame = None            # переменная для передачи текущего кадра с виртуальной камеры
boxes = None            # переменная для трансляции результатов
goal_point = None       # Координаты целевой точки
leftside_speed = 0      # скорость колес левого борта
rightside_speed = 0     # скорость колес правого борта
cam_num = 1             # текущая камера


# Запуск модулей в параллельных процессах
with ThreadPoolExecutor(max_workers=4) as executor:
    coppeliasim_task = executor.submit(coppeliasim_remoteAPI)
    objects_detecting = executor.submit(detect_objects)
    #crash_watchdog = executor.submit(collision_watchdog)
    robot_processing = executor.submit(robot_movement_proc)

    # Результаты работы модулей после остановки
    print(coppeliasim_task.result())
    print(objects_detecting.result())
    #print(crash_watchdog.result())
    print(robot_processing.result())
