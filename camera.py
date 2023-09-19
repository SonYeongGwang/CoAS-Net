import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs

class IntelCamera:
    def __init__(self):
        
        self.context = rs.context()
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = rs.align(rs.stream.color)
        self.spatial_filter = rs.spatial_filter()
        self.hole_filling_filter = rs.hole_filling_filter(0)

        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        self.device_product_line = str(device.get_info(rs.camera_info.product_line))

        print(self.device_product_line + " is ready")
        self.device_name = device.get_info(rs.camera_info.name).replace(" ", "_")
        self.device_name = self.device_name + "_" + device.get_info(rs.camera_info.serial_number)

        if self.device_product_line == 'L500':
            self.config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        else:
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        self.colorizer = rs.colorizer(color_scheme = 2)
        self.profile = self.pipeline.start(self.config)
    
    def stream(self):
        
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        ## filter depth frame
        # depth_frame = self.spatial_filter.process(depth_frame)
        # depth_frame = self.hole_filling_filter.process(depth_frame)

        colored_depth_frame = self.colorizer.colorize(depth_frame)

        self.color_image = np.asanyarray(color_frame.get_data())
        self.colored_depth_image = np.asanyarray(colored_depth_frame.get_data())
        self.depth_image = np.asanyarray(depth_frame.get_data())

        return self.color_image, self.depth_image

class KinectCamera(IntelCamera):
    def __init__(self):

        self.config = o3d.io.AzureKinectSensorConfig()
        self.sensor = o3d.io.AzureKinectSensor(self.config)
        self.device = 0

        if not self.sensor.connect(self.device):
            raise RuntimeError('Failed to connect to sensor')
        
        else:
            print('MicroSoft AzureKinect' + " is ready")
            self.device_product_line = 'AzureKinect'

    def stream(self):
        
        align_depth_to_color = True

        rgbd = self.sensor.capture_frame(align_depth_to_color)

        while rgbd is None:
            rgbd = self.sensor.capture_frame(align_depth_to_color)
            
        rgb = np.asarray(rgbd.color)
        self.color_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self.depth_image = np.asarray(rgbd.depth)

        return self.color_image, self.depth_image