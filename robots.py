import time
import numpy as np
import cv2


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
        self.goal_point = np.array([np.NAN, np.NAN])    # Целевая точка 
        self.__start_time = time.time()   # Время создания экземпляра
        self.__action = 0            # Флаг состояния робота
        self.__box_coords = np.array(
            [np.NAN, np.NAN, np.NAN, np.NAN])
        self.__keypoints_vec = np.array( # Вектор направления по ключевым точкам
            [np.NAN, np.NAN])
        self.__angle_keypoints = 0   # Угол отклонения по ключеввым точкам
        self.__angle_track = 0       # Угол отклонения по вектору движения
        self.__rotate_coef = 0       # Коэффициент подруливания в движении
        self.__history = np.array([[ # История перемещений робота
            time.time() - self.__start_time,   # Временная метка от старта
            self.__box_coords[0],    # Координата x трекинга
            self.__box_coords[1],    # Координата y трекинга
            self.leftside_speed,     #
            self.rightside_speed,    #
            self.__action,           #
            self.__angle_track,      #
            self.__angle_keypoints,  #
            self.goal_point[0],      # Координата x целевой точки
            self.goal_point[1],      # Координата y целевой точки
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
        for reverse in self.__history[::-1]:
            track = self.__history[-1][1:3], reverse[1:3]
            if np.linalg.norm(track[0] - track[1]) > 4:
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


    def __find_center(self, itemindex: np.array) -> tuple:
        '''Расчитываю центры контрольных точек робота'''
        try:
            y_min = np.min(itemindex[0][:])
            y_max = np.max(itemindex[0][:])
            x_min = np.min(itemindex[1][:])
            x_max = np.max(itemindex[1][:])
            return (x_min + x_max) // 2, (y_min + y_max) // 2
        except:
            return 0, 0
    

    def __find_keypoints(self, robot_roi: np.array) -> tuple:
        '''Нахожу ключевые точки робота по цветовым меткам'''
        # Ищу ключевую точку "головы" по желтой метке
        img_HSV = cv2.cvtColor(robot_roi, cv2.COLOR_RGB2HSV)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 255, 255])
        mask_yellow = cv2.inRange(img_HSV, lower_yellow, upper_yellow)
        x_head, y_head = self.__find_center(np.where(mask_yellow != 0))
        # Ищу ключевую точку "хвоста" по синей метке
        lower_blue = np.array([100, 150, 0])
        upper_blue = np.array([140, 255, 255])
        mask_blue = cv2.inRange(img_HSV, lower_blue, upper_blue)
        x_tail, y_tail = self.__find_center(np.where(mask_blue != 0))
        return x_head, y_head, x_tail, y_tail


    def find_keypoints_vec(self, frame) -> bool:
        ''''Определяем вектор направления по ключевым точкам'''
        # Переделать на поиск ключевых точек моделью YOLO keypoints
        try:
            # вырезаю робота по ограничительной рамке
            x, y, w, h = self.__box_coords
            # ищу ключевые точки по цветовым меткам
            robot_roi = frame[y-h//2:y+h//2, x-w//2:x+w//2]
            x_head, y_head, x_tail, y_tail = self.__find_keypoints(robot_roi)
            x_head = x_head + x - w // 2
            y_head = y_head + y - h // 2
            x_tail = x_tail + x - w // 2
            y_tail = y_tail + y - h // 2
            self.__keypoints_vec = np.array([[x_head, y_head], [x_tail, y_tail]])
            return True
        except Exception as e:
            print(e)
            self.__keypoints_vec = np.array([np.NAN, np.NAN])
            return False


    def find_itself(self, boxes) -> bool:
        '''Ищем в результатах обработки моделью кадра координаты робота'''
        # пока ищем по классу 0, потом нужно будет добавить идентификаторы
        try:
            robot_box = boxes[boxes.cls==0].cpu().numpy()
            self.__box_coords = robot_box.xywh[0].astype(dtype='int16')
            self.__add_history()
            return True
        except Exception as e:
            print(e)
            return False
    

    def check_cam(self, frame_size: tuple, cam) -> int:
        '''Проверяем нахождение в пограничной зоне переключения камеры'''
        # Проверяю условие для переключения соответствующей камеры
        if self.__box_coords[0] < frame_size[1] * 0.05:
            return 2
        elif self.__box_coords[0] > (frame_size[1] - frame_size[1] * 0.05):
            return 1
        else:
            return 0


    def find_way(self, frame, goal_point) -> bool:
        '''Определяем маршрут движения до целевой точки'''
        # Здесь будет обширный блок расчета маршрута минуя препятствия,
        # но пока это только заготовка для него.
        # Возвращает False если маршрут построить невозможно, либо
        # если робот достиг целевой точки
        if goal_point:
            if np.linalg.norm(self.__box_coords[0:2] - goal_point) > 20:
                self.goal_point = np.array(goal_point)
                return True
            else:
                self.goal_point = np.array([np.NAN, np.NAN])
                return False
                

    def robot_action(self) -> tuple:
        '''Робот движется по рассчитанному ранее маршруту'''
        # В момент изменения целевой точки поднимаем флаг
        # разворота на месте. Некрасиво, переделать этот блок
        if not np.array_equal(self.goal_point, self.__history[-1][-2:]):
                self.__angle_keypoints = self.__calc_angle(
                                        self.__keypoints_vec, self.goal_point)
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
                                        self.__keypoints_vec, self.goal_point)
        # Смотрим текущий знак угла отклонения от вектора к целевой точке
        sign = self.__calc_sign(self.__angle_keypoints)
        # Если знаки углов текущей и прошлой итерации отличаются,
        # значит робот перескочил через 0, флаг состояния переводится
        # в значение (1), то есть движение прямо
        if sign != self.__calc_sign(self.__history[-1][-3]):
            self.__action = 1
        # Скорость разворота замедляется в секторе +-15 градусов от goal_vector
        rotate_coef = 1 if abs(self.__angle_keypoints) < 15 else self.speed_limit
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
            self.goal_point[0],      # Координата x целевой точки
            self.goal_point[1],      # Координата y целевой точки
            ]], axis=0)
        # Если строк более 400, убираем первую
        if len(self.__history) > 400:
            self.__history = np.delete(self.__history, (0), axis=0)


    def show_history(self) -> None:
        '''Отображение истории'''
        print(self.__history)
