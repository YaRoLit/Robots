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
    Cam_Handle = sim.getObject('./Mast/Cam')
    # Запускаем симуляцию и цикл управления объектами сцены
    sim.startSimulation()
    #fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    #out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (800, 800))
    while run_flag:
        img, res = sim.getVisionSensorImg(Cam_Handle)
        img_RGB = np.frombuffer(img, dtype=np.uint8).reshape(res[1], res[0], 3)
        frame = img_RGB[::-1, :, :]
        img_BGR = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
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


def process_video() -> None:
    '''
    Отдельный процесс, отвечающий за обнаружение координат всех объектов в кадре
    с помощью предобученной модели и трансляцию в виде глобальных переменных.
    Должен работать на GPU.
    '''
    global frame
    global results
    global run_flag
    # Автоматический выбор устройства. Крайне рекомендуется иметь CUDA.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Загрузка модели YOLO
    model = YOLO('./Models/pose.pt')
    # Запуск цикла обработки видеопотока
    while run_flag:
        # Обрабатываем кадр, ищем объекты
        results = model(frame, device=device, verbose=False)


def crash_watchdog() -> None:
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
    global results
    global run_flag
    # Создаем экземпляр класса "робот", классы введены для поддержки нескольких
    # роботов на одной сцене, для каждого отдельного робота создается экземпляр
    robot_1 = robots.Robots(speed=4)
    # Переменная для счетчика кадров, в которых робот не обнаружен
    missed_frames_cnt = 0
    # Основной цикл управления роботом
    while run_flag:
        # Если робота "не видно" 10+ кадров, останавливаем его
        if missed_frames_cnt > 10:
            leftside_speed = 0
            rightside_speed = 0
        # Робот ищет свои координаты в результатах обработки кадра моделью
        if robot_1.find_itself(results=results):
            missed_frames_cnt = 0
        else:
            missed_frames_cnt += 1
            continue
        # Передаем роботу координаты целевой точки, строим маршрут
        # Если робот успешно построил маршрут, он двигается по нему
        if robot_1.find_way(goal_point):
            leftside_speed, rightside_speed = robot_1.robot_action()
            print(leftside_speed, rightside_speed)


run_flag = True         # флаг для остановки программы
frame = None            # переменная для передачи текущего кадра с виртуальной камеры
results = None          # переменная для трансляции результатов
goal_point = None       # Координаты целевой точки
leftside_speed = 0      # скорость колес левого борта
rightside_speed = 0     # скорость колес правого борта


# Запуск модулей в параллельных процессах
with ThreadPoolExecutor(max_workers=4) as executor:
    coppeliasim_task = executor.submit(coppeliasim_remoteAPI)
    video_processing = executor.submit(process_video)
    #crash_watchdog_ = executor.submit(crash_watchdog)
    robot_processing = executor.submit(robot_movement_proc)

    # Результаты работы модулей после остановки
    print(coppeliasim_task.result())
    print(video_processing.result())
    #print(crash_watchdog_.result())
    print(robot_processing.result())
