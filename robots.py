import time
import numpy as np


class Robots:
    def __init__(self,
                model='First',
                type='4X4_side',     # Колесный робот с бортовым приводом
                speed=1,
                ) -> None:
        self.model = model           # Модель робота
        self.type = type             # Тип робота (шасси)
        self.speed_limit = speed     # Скоростной лимит
        self.leftside_speed = 0      # Угловая скорость двигателей левого борта
        self.rightside_speed = 0     # Угловая скорость двигателей правого борта
        self.goal_point = None       # Целевая точка движения робота 
        self.__start_time = time.time()   # Время создания экземпляра
        self.__action = 0            # Флаг состояния робота
        self.__box_coords = None     # Координаты и размер ограничивающейй рамки
        self.__angle_track = 0       # Угол отклонения по вектору движения
        self.__rotate_coef = 0       # Коэффициент подруливания в движении
        self.__direction_vec = None     # Вектор направления робота по keypoints
        self.__angle_kp = 0          # Угол отклонения по контрольным точкам
        self.__history = np.array([[ # История перемещений робота
            time.time() - self.__start_time,   # Временная метка от старта
            np.NAN,                  # Координата x трекинга
            np.NAN,                  # Координата y трекинга
            self.leftside_speed,     #
            self.rightside_speed,    #
            self.__action,           #
            self.__angle_track,      #
            self.__angle_kp          #
        ]])


    def __calc_angle(self, dir_vec: tuple, goal_point: tuple) -> int:
        '''
        Вычисляем угол между вектором направления и вектором
        до целевой точки. Вектор направления передается целиком,
        его точка [1] является смежной, от которой строятся оба
        вектора.
        '''
        v_1 = np.array(dir_vec[1]) - np.array(dir_vec[0])
        v_2 = np.array(dir_vec[1]) - np.array(goal_point)
        angle = np.degrees(np.math.atan2(
                           np.linalg.det([v_1, v_2]),
                           np.dot(v_1, v_2)))
        
        return round(angle)


    def __check_distance(self, point_1: tuple, point_2: tuple) -> int:
        '''Вычислем длину вектора между двумя точками'''
        distance = ((point_1[0] - point_2[0]) ** 2 +
                    (point_1[1] - point_2[1]) ** 2) ** 0.5
        
        return int(distance)


    def __calc_boundingbox_midpoint(self) -> tuple:
        '''Отслеживаем изменения центральной точки ограничивающей рамки'''
        # Предполагалось, что трекинг будет по центральной точке рамки
        center_points = (
            ((self.__box_coords[0][0] + self.__box_coords[0][2]) // 2),
            ((self.__box_coords[0][1] + self.__box_coords[0][3]) // 2))

        return center_points


    def __calc_rotate_coef(self) -> tuple:
        '''Вычисляем коэффициент "подруливания" в движении'''
        # Эта проверка нужна чтобы избежать непонятных вылетов,
        # нужно разобраться в их причине и убрать её
        try:
            track = ((self.__history[-1][1:3]), (self.__history[-2][1:3]))
        # Проверяем, что смещение центральной точки не равно 0 (робот сместился)
            #if not np.array_equal(track[0], track[1]):
            if self.__check_distance(track[0], track[1]) > 1:
                # Определяем угол отклонения между вектором смещения и целевым
                self.__angle_track = self.__calc_angle(track, self.goal_point)
                angle = self.__angle_track if abs(self.__angle_track) > 45 else 45
                # Затем по углу определяем коэффициент подруливания
                self.__rotate_coef = (angle / 45) * self.speed_limit
        except Exception:
            return (0, 0)
        # В зависимости от знака коэффициента применяем его к нужному борту
        if self.__rotate_coef > 0:
            return (0, self.__rotate_coef)
        else:
            return (self.__rotate_coef, 0) 


    def __calc_sign(self, num: float) -> int:
        '''Определяем знак числа'''

        return -1 if num < 0 else 1


    def find_itself(self, results) -> bool:
        '''Ищем в результатах обработки моделью кадра координаты робота'''
        if results:
            for result in results:
                # Тут нужно сделать выбор конкретного класса робота по self.model
                # нужно подумать над введением идентификаторов для одинаковых моделей
                boxes = result.boxes
                keypoints = result.keypoints
            # Если вектор с контрольными точками не пустой
            if keypoints[0].shape == (1, 2, 2):
                self.__direction_vec = keypoints[0].xy.cpu().numpy().astype(np.int16)
                self.__box_coords = boxes.xyxy.cpu().numpy().astype(np.int16)
                return True
            else:
                return False
        else:
            print()    # Магический принт, без него не работает
            return False


    def find_way(self, goal_point: tuple) -> bool:
        '''Определяем маршрут движения до целевой точки'''
        # Здесь будет обширный блок расчета маршрута минуя препятствия,
        # но пока это только заготовка для него
        if goal_point:
            self.goal_point = goal_point
        else:
            return False

        return True


    def robot_action(self) -> tuple:
        '''Робот движется по рассчитанному ранее маршруту'''
        # Проверяем условия перехода в состояния:
        # - остановка (self.__action=0)
        # - прямолинейное движение с подруливанием (self.__action=1)
        # - разворот на месте (self.__action=-1)
        # ----------------------------------------------------------
        # Вычисляем угол отклонения от вектора до цели по ключевым точкам робота
        print(self.__angle_kp)
        self.__angle_kp = self.__calc_angle(self.__direction_vec[0],
                                            self.goal_point)
        # Если угол отклонения +-45 градусов, поднимаем флаг разворота на месте
        if abs(self.__angle_kp) > 45:
            self.__action = -1
        # Если флаг разворота поднят, разворачиваемся
        if self.__action == -1:
            self.robot_rotate()
        # Здесь проверяем условие достижения целевой точки и останавливаемся
        elif self.__check_distance(
            self.__direction_vec[0][1], self.goal_point) < 20:
            self.__action = 0
            self.robot_stop()
        # Если не разворот и не остановка, значит едем прямо
        else:
            self.__action = 1
            self.robot_move_straight()

        return self.leftside_speed, self.rightside_speed


    def robot_move_straight(self) -> None:
        '''Робот едет прямо'''
        # Определяем коэффициенты подруливания для каждого борта отдельно
        leftside_coef, rightside_coef = self.__calc_rotate_coef()
        self.leftside_speed = round((self.speed_limit + leftside_coef), 2)
        self.rightside_speed = round((-self.speed_limit + rightside_coef), 2)
        self.__add_history()

        return self.leftside_speed, self.rightside_speed


    def robot_rotate(self) -> None:
        '''Робот разворачивается на месте'''
        # Смотрим текущий знак угла отклонения от вектора к целевой точке
        sign = self.__calc_sign(self.__angle_kp)
        # Если знаки углов текущей и прошлой итерации отличаются,
        # значит робот перескочил через 0, флаг состояния переводится
        # в значение (1), то есть движение прямо
        if sign != self.__calc_sign(self.__history[-1][-1]):
            self.__action = 1
        # Скорость разворота замедляется в секторе +-45 градусов от goal_vector
        rotate_coef = 1 if abs(self.__angle_kp) < 45 else self.speed_limit
        self.leftside_speed = sign * rotate_coef
        self.rightside_speed = sign * rotate_coef
        self.__add_history()

        return self.leftside_speed, self.rightside_speed


    def robot_stop(self) -> None:
        '''Робот остановлен'''
        self.leftside_speed = 0
        self.rightside_speed = 0
        #print(self.__history[-1][0])

        return self.leftside_speed, self.rightside_speed
    

    def __add_history(self) -> None:
        '''Добавляем запись истории для каждой итерации движения и поворота'''
        # Делаем трекинг по keypoints[1]
        track = self.__direction_vec[0][1]
        self.__history = np.append(self.__history,[[
            time.time() - self.__start_time,    # Время от старта
            track[0],                # Координата x трекинга
            track[1],                # Координата y трекинга
            self.leftside_speed,     # Скорость двигателей левого борта
            self.rightside_speed,    # Скорость двигателей правого борта
            self.__action,           # Тип активности
            self.__angle_track,      # Угол по движению
            self.__angle_kp          # Угол по контрольным точкам
            ]], axis=0)
        # Если строк более 400, убираем первую
        if len(self.__history) > 400:
            self.__history = np.delete(self.__history, (0), axis=0)
