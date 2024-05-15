import time
import numpy as np
from ultralytics import YOLO


class Robots:
    def __init__(self, model: YOLO, speed: int=1) -> None:
        self.start_time         = time.time()
        self.model              = YOLO(model)
        self.speed_limit        = speed
        self.leftside_speed     = 0     #  1: forward; -1: backward
        self.rightside_speed    = 0     # -1: forward;  1: backward
        self.roi                = None
        self.goal_point         = np.array([])
        self.__action_flag      = 2     # Actions type:
                                        #-1: robot backward;
                                        # 0: robot stop;
                                        # 1: robot forward;
                                        # 2: robot rotate start;
                                        # 3: robot rotate;
                                        # 4: robot backward;
                                        # X (any different): robot stop.
        self.__track            = [None]
        self.__turn_time        = 0

    def find_itself(self, frame) -> bool:
        '''Поиск координат робота в кадре'''
        # Использование перехватчика ошибок вместо условия оказалось эффективнее
        try:
            # Если координаты "зоны внимания" не заданы при инициализации робота
            if self.roi is None:
                # Осуществляется поиск начального положения робота во всем кадре
                if not self.__find_roi(frame=frame):
                    # Если робот не найден, возврат пустого значения
                    return None
            # По координатам "зоны внимания" вырезается соответствующий фрагмент
            roi = frame[self.roi[1] : self.roi[1] + 640,
                        self.roi[0] : self.roi[0] + 640]
            # Обработка полученной "зоны внимания" при помощи модели YOLO
            results = self.model(roi, verbose=False, imgsz=640, max_det=1)
            if results:
                # Если робот в "зоне внимания" найден, трекинг его координаты
                coords = results[0].boxes[0].cpu().numpy().xywh[0].astype(dtype='int16')
                coords[:2] += self.roi    # x, y + roi
                self.__track[-1] = coords[:2]
                # Смещение "зоны внимания" в соответствии с движением робота
                if (frame.shape[0] - 320) > coords[1] > 320:
                    self.roi[1] = coords[1] - 320
                elif coords[1] < 320:
                    self.roi[1] = 0
                elif coords[1] >= (frame.shape[0] - 320):
                    self.roi[1] = frame.shape[0] - 640
                if (frame.shape[1] - 320) > coords[0] > 320:
                    self.roi[0] = coords[0] - 320
                elif coords[0] < 320:
                    self.roi[0] = 0
                elif coords[0] >= (frame.shape[1] - 320):
                    self.roi[0] = frame.shape[1] - 640
                return True
        except Exception as e:
            # Обработчик ошибок и их вывод
            print('!!!!---Ошибка обработки кадра моделью---!!!!', e)
            return False

    def check_gp(self, goal_point: tuple) -> bool:
        '''Проверка достижения целевой точки'''
        goal_point = np.array(goal_point)    # необходимо для расчета дистанции
        # Факт достижения целевой точки определяется по условию приближения
        # центральной точки ограничивающей рамки к целевой точки на некоторое
        # расстояние, в данном случае менее 20
        if np.linalg.norm(self.__track[-1] - goal_point) < 20:
            # Когда очередная точка достигнута, активен флаг разворота на месте
            self.__action_flag = 2
            # возврат False для удаления точки из маршрута
            return False
        # если до точки > 20, её координаты становятся текущей целью робота
        self.goal_point = goal_point
        return True

    def action(self) -> tuple:
        '''Определение типа активности робота'''
        # Это действие необходимо при инициализации, чтобы трек был не нулевой
        if len(self.__track) < 2:
            self.__add_track()
        # Вызов соответствующего метода в зависимости от значения флага
        if self.__action_flag == 1:
            return self.moving_forward()
        elif self.__action_flag == 2:
            return self.rotation_start()
        elif self.__action_flag == 3:
            return self.rotation()
        else:
            return (0, 0)    # robot stop

    def moving_forward(self) -> tuple:
        '''Робот едет вперед'''
        # Проверяется длина смещения робота от предыдущей точки трека
        # В том случае, если координаты точки остались прежними, по
        # ним не строится вектор смещения и они не попадают в трек
        if self.__calc_displacement() > 1:    # длина смещения
            # рассчитывается коэффициент подруливания для корректировки курса
            leftside_coef, rightside_coef = self.__calc_rotate_coef()
            # определяется скорость двигателей с учетом коэффициента
            self.leftside_speed = round((self.speed_limit - leftside_coef), 2)
            self.rightside_speed = round((-self.speed_limit + rightside_coef), 2)
            # точка сохраняется в трек
            self.__add_track()
        # возврат числовых значений работы двигателей для управления роботом
        return self.leftside_speed, self.rightside_speed

    def rotation_start(self) -> tuple:
        '''Робот начинает выполнять разворот на месте'''
        # здесь определяется угол до целевой точки от текущего направления
        # и его знак, чтобы крутить в нужную сторону
        angle = self.__calc_angle(self.__track[-2:], self.goal_point)
        angle_sign = self.__calc_sign(angle)
        # устанавливается скорость и направление вращения бортовых двигателей
        self.leftside_speed = self.speed_limit * angle_sign
        self.rightside_speed = -self.rightside_speed * angle_sign
        # определяется длительность разворота в секундах
        turn_time = self.__calc_rotate_time(angle)
        # после необходимых расчетов перевод флага в состояние "разворот"
        self.__action_flag = 3
        # определение временной отметки, до которой продолжать разворот
        self.__turn_time = turn_time + time.time()
        # возврат числовых значений работы двигателей для управления роботом
        return self.leftside_speed, self.rightside_speed

    def rotation(self) -> tuple:
        '''Робот разворачивается на месте'''
        # Пока не достигнута определенная ранее временная отметка
        # робот продолжает разворачиваться на месте
        if time.time() < self.__turn_time:
            return self.leftside_speed, self.rightside_speed
        # После достижения отметки активируется флаг "движение вперед"
        else:
            self.__action_flag = 1
            return (0, 0)    # robot stop

    def __add_track(self) -> bool:
        '''Добавление точки в трек'''
        self.__track.append(self.__track[-1])
        #self.__track = self.__track[-100:]           

    def __find_roi(self, frame) -> bool:
        '''Определение roi при инициализации сцены'''
        # Координаты робота определяются на "полном кадре" с камеры
        # после чего фиксируется нулевая точка (вержний левый угол)
        # зоны внимания путём вычитания из центральной точки размеров рамки
        if frame is not None:
            results = self.model(frame, verbose=False, max_det=1)
            if results:
                rob_box = results[0].boxes[0].cpu().numpy().xywh[0].astype(dtype='int16')
                self.roi = [rob_box[0] - rob_box[2], rob_box[1] - rob_box[3]]
                return True
        return False

    def __calc_angle(self, vec: np.array, point: np.array) -> int:
        '''Расчет угла между вектором и точкой'''
        v_1 = np.array(vec[0]) - np.array(vec[1])
        v_2 = np.array(vec[1]) - np.array(point)
        angle = np.degrees(np.math.atan2(np.linalg.det([v_1, v_2]),
                                                np.dot(v_1, v_2)))
        return round(angle)

    def __calc_displacement(self) -> float:
        '''Расчет длины смещения центральной точки ограничивающей рамки'''
        return np.linalg.norm(self.__track[-1] - self.__track[-2])

    def __calc_rotate_coef(self) -> tuple:
        '''Расчет коэффициента подруливания в движении'''
        # Определение угла отклонения между вектором движения и целевым
        angle = self.__calc_angle(self.__track[-2:], self.goal_point)
        # Подруливающий коэффициент рассчитывается в зависимости от угла
        if abs(angle) > 45:
            rotate_coef = self.speed_limit * self.__calc_sign(angle)
        else:
            rotate_coef = round(angle / 45 * self.speed_limit, 2)
        if rotate_coef > 0:
            return (0, abs(rotate_coef))
        else:
            return (abs(rotate_coef), 0)

    def __calc_rotate_time(self, angle: int) -> float:
        '''Расчет необходимого времени разворота на месте'''
        # Используется фиксированная поправка, переделать на динамическую!!!!!!!
        turn_time = abs(angle) * (1 / self.speed_limit) * 0.07
        return turn_time

    def __calc_sign(self, num: float) -> int:
        '''Определение знака числа'''
        return -1 if num < 0 else 1

    def show_history(self) -> None:
        '''Отображение истории'''
        print(time.time() - self.start_time)
        return self.__track
