import time
import numpy as np
import cv2
import torch
from ultralytics import YOLO


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Robots:
    def __init__(self,
                model='./Models/pose.pt',
                robot_type=0,
                speed=1,
                start_place=[640, 320]    # roi_zero_point [x, y]
                ) -> None:
        self.model = YOLO(model)           # Файл модели робота
        self.robot_type = robot_type    # Тип робота
        self.speed_limit = speed     # Скоростной лимит
        self.leftside_speed = 0      # Угловая скорость двигателей левого борта
        self.rightside_speed = 0     # Угловая скорость двигателей правого борта
        self.goal_point = np.array([[np.NAN, np.NAN], ])    # Массив целевых точек (маршрут) 
        self.__start_time = time.time()   # Время создания экземпляра
        self.__action = 0            # Флаг состояния робота
        self.__box_coords = np.array(
            [np.NAN, np.NAN, np.NAN, np.NAN])
        self.__keypoints_vec = np.array( # Вектор направления по ключевым точкам
            [np.NAN, np.NAN])
        self.__angle_keypoints = 0   # Угол отклонения по ключеввым точкам
        self.__angle_track = 0       # Угол отклонения по вектору движения
        self.__rotate_coef = 0       # Коэффициент подруливания в движении
        self.roi = start_place
        self.__history = np.array([[ # История перемещений робота
            time.time() - self.__start_time,   # Временная метка от старта
            self.__box_coords[0],    # Координата x трекинга
            self.__box_coords[1],    # Координата y трекинга
            self.leftside_speed,     #
            self.rightside_speed,    #
            self.__action,           #
            self.__angle_track,      #
            self.__angle_keypoints,  #
            self.goal_point[0][0],      # Координата x целевой точки
            self.goal_point[0][1],      # Координата y целевой точки
        ]])


    def __calc_angle(self, dir_vec: np.array, goal_point: np.array) -> int:
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


    def __calc_rotate_coef(self) -> tuple:
        '''Вычисляем коэффициент "подруливания" в движении'''
        # Ищем координаты такой предыдущей точки в истории,
        # расстояние до которой от текущей больше 4 пикселей.
        # Затем строим между ними вектор и вычисляем угол отклонения.
        for reverse in self.__history[::-1]:
            track = self.__history[-1][1:3], reverse[1:3]
            if np.linalg.norm(track[0] - track[1]) > 2:
                self.__angle_track = self.__calc_angle(track, self.goal_point[0])
                break
        # Если отклонение больше 45 градусов, крутим по полной.
        if abs(self.__angle_track) > 45:
            self.__rotate_coef = self.speed_limit * self.__calc_sign(
                                                            self.__angle_track)
        # Если меньше 45, то варьируем скорость подруливания от угла.
        else:
            self.__rotate_coef = round(
                                 self.__angle_track / 45 * self.speed_limit, 2)
        # Применяем коэффициент подруливания к соотствующему борту от знака.
        if self.__rotate_coef > 0:
            return (0, abs(self.__rotate_coef))
        else:
            return (abs(self.__rotate_coef), 0)


    def __calc_sign(self, num: float) -> int:
        '''Определяем знак числа'''
        return -1 if num < 0 else 1


    def find_itself(self, frame) -> bool:
        '''Нахожу координаты робота в кадре. Двигаю окно roi.'''
        try:
            roi = frame[self.roi[1] : self.roi[1] + 640,
                        self.roi[0] : self.roi[0] + 640]
            results = self.model(roi, device=device, verbose=False, imgsz=640, max_det=1)
            if results:
                self.__keypoints_vec = results[0].keypoints[0].cpu().numpy().xy[0].astype(dtype='int16')
                self.__keypoints_vec[0] += self.roi
                self.__keypoints_vec[1] += self.roi
                self.__box_coords = results[0].boxes[0].cpu().numpy().xywh[0].astype(dtype='int16')
                self.__box_coords[0] += self.roi[0]
                self.__box_coords[1] += self.roi[1]
                print(self.__box_coords, self.roi[0], self.roi[1], self.__angle_keypoints)
                if (frame.shape[0] - 320) > self.__box_coords[1] > 320:
                    self.roi[1] = self.__box_coords[1] - 320
                elif self.__box_coords[1] < 320:
                    self.roi[1] = 0
                elif self.__box_coords[1] >= (frame.shape[0] - 320):
                    self.roi[1] = frame.shape[0] - 640
                if (frame.shape[1] - 320) > self.__box_coords[0] > 320:
                    self.roi[0] = self.__box_coords[0] - 320
                elif self.__box_coords[0] < 320:
                    self.roi[0] = 0
                elif self.__box_coords[0] >= (frame.shape[1] - 320):
                    self.roi[0] = frame.shape[1] - 640
                self.__add_history()
                return roi    # True
        except Exception as e:
            print(e)
            return None    # False
    

    def check_cam(self, frame_size: tuple, cam) -> int:
        '''Проверяем нахождение в пограничной зоне переключения камеры'''
        # Проверяю условие для переключения соответствующей камеры
        pass


    def check_way(self, goal_point) -> bool:
        '''Определяем маршрут движения до целевой точки'''
        # Если появилась контрольная точка (not None)
        if goal_point:
            # Если маршрут "пустой"
            if np.isnan(self.goal_point[0][0]):
                # Пытаемся создать маршрут, возвращаем True при успехе
                return self.create_way(goal_point)
            # Если маршрут "не пустой"
            else:
                # Проверяем достижение текущей [0] контрольной точки
                if np.linalg.norm(self.__box_coords[0:2] - self.goal_point[0]) < 20:
                    # В случае достижения, удаляем её
                    self.goal_point = np.delete(self.goal_point, (0), axis=0)
                    # Если после удаления не осталось активных точек, возвращаем False
                    if np.isnan(self.goal_point[0][0]):
                        return False
                    # Если активные точки ещё остались, возвращаем True
                    else:
                        return True
                # Пока текущая контрольная точка не достигнута, возвращаем True
                else:
                    return True
    

    def create_way(self, goal_point) -> bool:
        '''Создаем маршрут (перечень контрольных точек) для следования'''
        self.goal_point = np.vstack((goal_point, self.goal_point))
        return True


    def robot_action(self) -> tuple:
        '''Робот движется по рассчитанному ранее маршруту'''
        # В момент изменения целевой точки поднимаем флаг
        # разворота на месте. Некрасиво, переделать этот блок
        if not np.array_equal(self.goal_point[0], self.__history[-1][-2:]):
                self.__angle_keypoints = self.__calc_angle(
                                        self.__keypoints_vec, self.goal_point[0])
                self.__add_history()
                self.__action = 2
        # Если поднят флаг разворота на месте, разворачиваемся
        if self.__action == 2:
            return self.robot_rotation()
        else:
            return self.robot_move_forward()


    def robot_rotation(self) -> tuple:
        '''Робот разворачивается на месте'''
        self.__angle_keypoints = self.__calc_angle(
                                        self.__keypoints_vec, self.goal_point[0])
        # Смотрим текущий знак угла отклонения от вектора к целевой точке
        sign = self.__calc_sign(self.__angle_keypoints)
        # Если знаки углов текущей и прошлой итерации отличаются,
        # значит робот перескочил через 0, флаг состояния переводится
        # в значение (1), то есть движение прямо
        if sign != self.__calc_sign(self.__history[-1][-3]):
            self.__action = 1
        # Скорость разворота замедляется в секторе +-15 градусов от goal_vector
        rotate_coef = 1 if abs(self.__angle_keypoints) < 22 else self.speed_limit
        self.leftside_speed = sign * rotate_coef
        self.rightside_speed = sign * rotate_coef
        return self.leftside_speed, self.rightside_speed


    def robot_move_forward(self) -> tuple:
        '''Робот едет вперед'''
        leftside_coef, rightside_coef = self.__calc_rotate_coef()
        self.leftside_speed = round((self.speed_limit - leftside_coef), 2)
        self.rightside_speed = round((-self.speed_limit + rightside_coef), 2)
        return self.leftside_speed, self.rightside_speed


    def robot_move_backward(self) -> tuple:
        '''Робот едет назад'''
        # Пока только заготовка, не работает
        self.leftside_speed, self.rightside_speed = -1, 1
        return self.leftside_speed, self.rightside_speed


    def robot_stop(self) -> tuple:
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
            self.__angle_keypoints,
            self.goal_point[0][0],      # Координата x целевой точки
            self.goal_point[0][1],      # Координата y целевой точки
            ]], axis=0)
        # Если строк более 400, убираем первую
        if len(self.__history) > 10000:
            self.__history = np.delete(self.__history, (0), axis=0)


    def show_history(self) -> None:
        '''Отображение истории'''
        return self.__history
