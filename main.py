# стандартные модули
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import cv2
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from screeninfo import get_monitors
# пользовательские модули
import robots


monitor = get_monitors()[0]


def transform_frame(frame: np.array, resize_ratio: int)-> None:
    '''
    Функция трансформации кадра для корректного отображения
    '''
    # Изменение размеров кадра
    out_width = round(monitor.width / resize_ratio)
    out_height = round(monitor.height / resize_ratio)
    # Изменение цветовой схемы для CV2
    frame_cv2 = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # Отображение текущей целевой точки
    if len(path):
        cv2.circle(frame_cv2, (path[0][0], path[0][1]), 20, (0, 0, 255), -1)
    resize_frame = cv2.resize(frame_cv2, (out_width, out_height))
    return resize_frame


def coppeliasim_remoteAPI()-> None:
    '''
    Отдельный поток взаимодействия с объектами сцены и трансляции видео
    '''
    # Глобальные переменные, необходимые для взаимодействия между потоками
    global frame                    # --> для передачи кадра с видеокамеры
    global leftside_speed           # <-- для получения параметров скорости ++
    global rightside_speed          # ++ вращения колес левого и правого бортов
    global run_flag                 # флаг глобального останова всех потоков
    global path                     # <-- для получения массива точек маршрута
    # Инициализация клиента CoppeliaSim, подключение всех объектов сцены
    client = RemoteAPIClient()
    sim = client.require('sim')
    sim.loadScene(sim.getStringParam(
        sim.stringparam_scenedefaultdir) + '/test.ttt')         # файл сцены
    robot = sim.getObject('./RobotnikSummitXL')                 # объект робота
    frontleft_motor = sim.getObject('./front_left_wheel')       # объекты всех --
    frontright_motor = sim.getObject('./front_right_wheel')     # -- бортовых --
    backleft_motor = sim.getObject('./back_left_wheel')         # -- приводов --
    backright_motor = sim.getObject('./back_right_wheel')       # -- робота
    cam_handle = sim.getObject(f'./cam_1')                      # видеокамера
    sim.startSimulation()
    # Запуск цикла взаимодействия с объектами сцены
    while run_flag:
        # Захват кадра с виртуальной камеры
        img, res = sim.getVisionSensorImg(cam_handle)
        # Перевод в RGB формат
        img_RGB = np.frombuffer(img, dtype=np.uint8).reshape(res[1], res[0], 3)
        frame = img_RGB[::-1, :, :]
        # Определение позиции робота, если необходим его трекинг ч\з CoppeliaSim
        pos = sim.getObjectPosition(robot)
        # Изменение размера кадра для корректного отображения
        cv2_frame = transform_frame(frame, resize_ratio=1.5)
        # Отрисовка кол-ва оставшихся точек маршрута в виде текста
        if len(path):
            cv2.putText(cv2_frame, f'Goal points left {len(path)}',
                                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                                   (255, 0, 0), 2, cv2.LINE_AA)
        # Вывод изображения сцены с внешней видеокамеры
        cv2.imshow('scene', cv2_frame)
        # Управление двигателями робота
        sim.setJointTargetVelocity(frontleft_motor, leftside_speed)
        sim.setJointTargetVelocity(frontright_motor, rightside_speed)
        sim.setJointTargetVelocity(backleft_motor, leftside_speed)
        sim.setJointTargetVelocity(backright_motor, rightside_speed)
        # Проверка нажатия "Esc", которое останавливает все потоки и программу
        if cv2.waitKey(10) & 0xFF == 27: 
            run_flag = False
            break
    # После выхода из цикла останавливается сцена и уничтожаются все окна
    sim.stopSimulation()
    cv2.destroyAllWindows()


def robot_movement_proc()-> None:
    '''
    Управление роботом (в отдельном потоке это сделано для того,
    чтобы можно было управлять несколькими различными роботами на
    одной сцене одновременно, для каждого из них нужен такой же поток)
    '''
    global frame                    # <-- для получения кадра с видеокамеры
    global leftside_speed           # --> для передачи параметров скорости\
    global rightside_speed          # вращения колес левого и правого бортов
    global run_flag                 # флаг глобального останова всех потоков
    global path                     # <-- для получения массива точек маршрута
    # Создаем экземпляр робота, определяем параметры (файл модели и скорость)
    # другие модели есть в папке Models: pose8n80.pt, detect9c50.pt, etc.
    robot_1 = robots.Robots(model='./Models/detect8n80.pt', speed=8)
    # Переменная для счетчика кадров, в которых робот не обнаружен
    missed_frames_cnt = 0
    # Основной цикл управления роботом
    while run_flag:
        # Если робота "не видно" 10+ кадров, вращаем его до сизого дыма
        if missed_frames_cnt > 10:
            leftside_speed, rightside_speed = 1, 1
        # Робот ищет себя в кадре
        if robot_1.find_itself(frame=frame):
            missed_frames_cnt = 0
        else:
            # пока он себя не обнаружил, прибавляется счетчик пропущенных кадров
            missed_frames_cnt += 1
            # и управляющий цикл дальне не проходит, а стартует по новой
            continue
        # Если робот нашел себя, ему передаются координаты текущей целевой точки
        if robot_1.check_gp(goal_point=path[0]):
            # трансляция параметров работы бортовых двигателей для потока сцены
            leftside_speed, rightside_speed = robot_1.action()
        # Если робот достиг целевой точки или маршрут невозможен,
        # то останов робота и сброс целевой точки
        else:
            leftside_speed, rightside_speed = (0, 0)
            # из маршрута удаляется пройденная целевая точка
            path = path[1:]
        if len(path) == 0:
            run_flag = False
            break
    # При необходимости можно получить трекинг робота по данным модели
    track = np.array(robot_1.show_history())
    # И сохранить его в виде датафрейма Pandas для анализа
    df = pd.DataFrame({'x': track[:, 0], 'y': track[:, 1]})
    #df.to_csv('track.csv', index=False)


if __name__ == '__main__':
    ''' Запуск скрипта через терминал'''
    # Глобальные переменные, необходимые для взаимодействия между
    # отдельными независимыми потоками для роботов и сцены
    run_flag        = True      # флаг для останова программы
    frame           = None      # текущий кадр с виртуальной камеры
    leftside_speed  = 0         # скорость колес левого борта
    rightside_speed = 0         # скорость колес правого борта
    path            = [[150, 150],    # маршрут в виде списка--
                       [1500, 300],   # --целевых точек--
                       [200, 1500],   # --либо можно в виде--
                       [1600, 1500],  # --подгружаемого файла--
                       [300, 300]]    # --массива точек numpy
    #path            = np.load('path.npy')
    # Запуск модулей в отдельных параллельных асинхронных потоках
    with ThreadPoolExecutor(max_workers=4) as executor:
        coppeliasim_task = executor.submit(coppeliasim_remoteAPI)
        robot_processing = executor.submit(robot_movement_proc)
        # Вывод результатов работы модулей после останова
        print(coppeliasim_task.result())
        print(robot_processing.result())
