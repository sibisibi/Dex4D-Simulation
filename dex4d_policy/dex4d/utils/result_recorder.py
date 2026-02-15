import os
from isaacgym import gymapi
import numpy as np
import cv2
import time

class VideoRecorder:
    def __init__(self, env, model_dir, height=480, width=640, fps=60, env_id=0):
        self.env = env
        self.model_dir = model_dir
        self.video_dir = os.path.dirname(model_dir)
        model_name = model_dir.split('/')[-2]
        model_iter = os.path.basename(self.model_dir).split('model_')[-1].split('.')[0]
        self.height = height
        self.width = width
        self.fps = fps
        self.env_id = env_id

        camera_properties = gymapi.CameraProperties()
        camera_properties.width = self.width
        camera_properties.height = self.height
        self.cam = self.env.gym.create_camera_sensor(self.env.envs[self.env_id], camera_properties)

        camera_offset = gymapi.Vec3(1, 0, 1)
        camera_target = gymapi.Vec3(0, 0, 0.5)
        self.env.gym.set_camera_location(self.cam, self.env.envs[self.env_id], camera_offset, camera_target)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_path = os.path.join(self.video_dir, f'{model_name}_iter_{model_iter}_env_{self.env_id}.mp4')
        self.video_writer = cv2.VideoWriter(self.video_path, fourcc, self.fps, (self.width, self.height))

    def record_frame(self):
        self.env.gym.fetch_results(self.env.sim, True)
        self.env.gym.step_graphics(self.env.sim)
        self.env.gym.render_all_camera_sensors(self.env.sim)
        img = self.env.gym.get_camera_image(self.env.sim, self.env.envs[self.env_id], self.cam, gymapi.IMAGE_COLOR)
        img = np.reshape(img, (self.height, self.width, 4))
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        self.video_writer.write(img[..., :3])
        
        
class MetricWriter:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.model_name = model_dir.split('/')[-2]
        self.model_iter = os.path.basename(self.model_dir).split('model_')[-1].split('.')[0]
        self.time_stamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

    def write_metrics_txt(self, metrics_dict):
        metric_path = os.path.join(os.path.dirname(self.model_dir), f'{self.model_name}_iter_{self.model_iter}_metrics.txt')
        metrics_dict = {'time_stamp': self.time_stamp, **metrics_dict}
        with open(metric_path, 'a') as f:
            metrics_str = ', '.join([f'{key}: {value}' for key, value in metrics_dict.items()])
            f.write(metrics_str + '\n')
            f.flush()
    
    def write_metrics_csv(self, metrics_dict):
        metric_path = os.path.join(os.path.dirname(self.model_dir), f'{self.model_name}_iter_{self.model_iter}_metrics.csv')
        metrics_dict = {'time_stamp': self.time_stamp, **metrics_dict}
        file_exists = os.path.isfile(metric_path)
        with open(metric_path, 'a') as f:
            if not file_exists:
                header = ','.join(metrics_dict.keys())
                f.write(header + '\n')
            values = ','.join([str(value) for value in metrics_dict.values()])
            f.write(values + '\n')
            f.flush()