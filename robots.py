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
        self.goal_point = (np.NAN, np.NAN)    # Целевая точка движения робота 
        self.__start_time = time.time()   # Время создания экземпляра
        self.__action = 0            # Флаг состояния робота
        self.__box_coords = None     # Координаты и размер ограничивающейй рамки
        self.__angle_track = 0       # Угол отклонения по вектору движения
        self.__rotate_coef = 0       # Коэффициент подруливания в движении
        self.__history = np.array([[ # История перемещений робота
            time.time() - self.__start_time,   # Временная метка от старта
            np.NAN,                  # Координата x трекинга
            np.NAN,                  # Координата y трекинга
            self.leftside_speed,     #
            self.rightside_speed,    #
            self.__action,           #
            self.__angle_track,      #
            self.goal_point[0],      # Координата x целевой точки
            self.goal_point[1],      # Координата y целевой точки
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


    def __calc_rotate_coef(self) -> tuple:
        '''Вычисляем коэффициент "подруливания" в движении'''
        for reverse in self.__history[::-1]:
            track = self.__history[-1][1:3], reverse[1:3]
            if self.__check_distance(track[0], track[1]) > 4:
                self.__angle_track = self.__calc_angle(track, self.goal_point)
                break
        if abs(self.__angle_track) > 45:
            self.__rotate_coef = self.speed_limit * self.__calc_sign(self.__angle_track)
        else:
            self.__rotate_coef = round(self.__angle_track / 45 * self.speed_limit, 2)
        if self.__rotate_coef > 0:
            return (0, abs(self.__rotate_coef))
        else:
            return (abs(self.__rotate_coef), 0)


    def __calc_sign(self, num: float) -> int:
        '''Определяем знак числа'''

        return -1 if num < 0 else 1


    def find_itself(self, boxes) -> bool:
        '''Ищем в результатах обработки моделью кадра координаты робота'''
        # пока ищем по классу 0, потом нужно будет добавить идентификаторы
        try:
            robot_box = boxes[boxes.cls==0].cpu().numpy()
            self.__box_coords = robot_box.xywh[0]
            self.__add_history()
            return True
        except Exception as e:
            print(e)
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
        if self.goal_point != (self.__history[-1][-2], self.__history[-1][-1]):
            # Проверка, что целевая точка изменилась для разворота на месте
            # Только для типов робота с бортовым приводом, по keypoints
            pass
        if self.__check_distance(
            self.__box_coords, self.goal_point) < 20:
            self.__action = 0
            return self.robot_stop()
        else:
            self.__action = 1
            return self.robot_move_forward()


    def robot_move_forward(self) -> None:
        '''Робот едет вперед'''
        leftside_coef, rightside_coef = self.__calc_rotate_coef()
        self.leftside_speed = round((self.speed_limit - leftside_coef), 2)
        self.rightside_speed = round((-self.speed_limit + rightside_coef), 2)

        return self.leftside_speed, self.rightside_speed


    def robot_move_backward(self) -> None:
        '''Робот едет назад'''
        # Пока только заготовка, не работает
        pass

        return self.leftside_speed, self.rightside_speed


    def robot_stop(self) -> None:
        '''Робот остановлен'''
        self.leftside_speed = 0
        self.rightside_speed = 0

        return self.leftside_speed, self.rightside_speed
    

    def __add_history(self) -> None:
        '''Добавляем запись истории для каждой итерации движения и поворота'''
        self.__history = np.append(self.__history,[[
            time.time() - self.__start_time,    # Время от старта
            self.__box_coords[0],    # Координата x трекинга
            self.__box_coords[1],    # Координата y трекинга
            self.leftside_speed,     # Скорость двигателей левого борта
            self.rightside_speed,    # Скорость двигателей правого борта
            self.__action,           # Тип активности
            self.__angle_track,      # Угол по движению
            self.goal_point[0],      # Координата x целевой точки
            self.goal_point[1],      # Координата y целевой точки
            ]], axis=0)
        # Если строк более 400, убираем первую
        if len(self.__history) > 400:
            self.__history = np.delete(self.__history, (0), axis=0)
