import numpy as np
import cv2
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from screeninfo import get_monitors


class coppeliasim_remoteAPI:
    def __init__(self, scene_name: str='/base.ttt') -> None:
        '''asa'''
        self.handle_mode = True
        client = RemoteAPIClient()
        self.sim = client.require('sim')
        self.sim.loadScene(self.sim.getStringParam(
            self.sim.stringparam_scenedefaultdir) + scene_name)
        self.frontleft_motor = self.sim.getObject('./front_left_wheel')
        self.frontright_motor = self.sim.getObject('./front_right_wheel')
        self.backleft_motor = self.sim.getObject('./back_left_wheel')
        self.backright_motor = self.sim.getObject('./back_right_wheel')
        cam_handle_0 = self.sim.getObject('./cam_0')
        cam_handle_1 = self.sim.getObject('./cam_1')
        self.cameras = [cam_handle_0, cam_handle_1]
        self.sim.startSimulation()
    
    def get_frame(self, cam_num: int=0) -> np.array:
        '''asx'''
        img, res = self.sim.getVisionSensorImg(self.cameras[cam_num])
        img_RGB = np.frombuffer(img, dtype=np.uint8).reshape(res[1], res[0], 3)
        self.frame = img_RGB[::-1, :, :]
        return self.frame
    
    def transform_cv2_frame(self, frame: np.array, resize_ratio: int=2) -> np.array:
        '''fgh'''   
        monitor = get_monitors()[0]
        out_width = round(monitor.width / resize_ratio)
        out_height = round(monitor.height / resize_ratio)
        resize_frame = cv2.resize(frame, (out_width, out_height))
        return cv2.cvtColor(resize_frame, cv2.COLOR_RGB2BGR)

    def stop(self) -> None:
        '''asf'''
        self.sim.stopSimulation()
        cv2.destroyAllWindows()
    
    def handle_mode_movement(self) -> tuple:
        '''asd'''
        key = cv2.waitKey(10)
        if key == ord('w'):
            return 8, -8
        elif key == ord('s'):
            return -8, 8
        elif key == ord('a'):
            return -8, -8
        elif key == ord('d'):
            return 8, 8
        return 0, 0
    
    def update_motors(self, leftside_speed: int, rightside_speed: int) -> None:
        '''fgf'''
        self.sim.setJointTargetVelocity(self.frontleft_motor, leftside_speed)
        self.sim.setJointTargetVelocity(self.frontright_motor, rightside_speed)
        self.sim.setJointTargetVelocity(self.backleft_motor, leftside_speed)
        self.sim.setJointTargetVelocity(self.backright_motor, rightside_speed)


if __name__ == '__main__':
    copsim_scene = coppeliasim_remoteAPI(scene_name='/base_2cam.ttt')
    while True:
        frame = copsim_scene.get_frame()
        cv2_frame = copsim_scene.transform_cv2_frame(frame, resize_ratio=2)
        cv2.imshow('stream', cv2_frame)
        if copsim_scene.handle_mode:
            leftside_speed, rightside_speed = copsim_scene.handle_mode_movement()
            copsim_scene.update_motors(leftside_speed, rightside_speed)
        if cv2.waitKey(10) & 0xFF == 27: 
            copsim_scene.stop()
            break