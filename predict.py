import argparse
import numpy as np
import torch
import time
import cv2
import copy
import sys
import glob

from utils import remove_module, array_to_tensor, duplicate_channel_cv

from model import SuctionModel
from camera import IntelCamera, KinectCamera

parser = argparse.ArgumentParser()
parser.add_argument('--mode', help='Mode to predict either "dataset" or "streaming"', default='dataset', type=str)
parser.add_argument('--camera_model', default='D400', help="L515, D400, Kinect")
parser.add_argument('--example_modality', help='Select modality of the example; "sim" or "real"', default='sim')

## parser arguments
in_args = parser.parse_args(sys.argv[1:])
checkpoint_path = "./checkpoint"

def process_prediction_map(out):
    prediction_map = out[0][1,:,:].cpu().detach().numpy()
    prediction_map = np.where(prediction_map < 0, 0, prediction_map)
    prediction_map = cv2.GaussianBlur(prediction_map, (25, 25), 5)

    return prediction_map

def get_suction_point(prediction_map):
    max_pred = np.max(prediction_map)
    suction_point = np.where(prediction_map==max_pred)
    
    return suction_point, max_pred

def prediction_to_heatmap(prediction_map, max_pred, camera):
    prediction_map = (prediction_map*(255/max_pred)).astype(np.uint8)
    org_prediction_map = copy.copy(prediction_map)
    prediction_map = duplicate_channel_cv(prediction_map, camera=camera)
    heatmap = cv2.applyColorMap(prediction_map, cv2.COLORMAP_JET)

    return org_prediction_map, heatmap

## load model
device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
print("INFO: Predicting with {}".format(device))

t = time.time()
print("INFO: Loading the model and pretrained parameter...")

model = SuctionModel(backbone_name="resnet101", pretrained_backbone=True)
state_dict = torch.load(checkpoint_path+"/coas-net_checkpoint_full_method/suction_net_check_point_best.pt", map_location=device)
print("INFO: Loading the model and pretrained parameter took {:.3f}s".format(time.time()-t))

t = time.time()
print("INFO: Loading the model to the {}...".format(device))
model = model.to(device=device)
print("INFO: Loading the model to the {} took {:.3f}s".format(device, time.time()-t))

t = time.time()
print("INFO: Loading the pretrained prameters")
state_dict = remove_module(state_dict)
model.load_state_dict(state_dict)
print("INFO: Loading the pretrained prameters took {:.3f}s".format(time.time()-t))

model.eval()

if in_args.mode == 'streaming':
    if in_args.camera_model == "D400" or in_args.camera_model == "L515" :
        cam = IntelCamera()
    elif in_args.camera_model == "Kinect":
        cam = KinectCamera()
    else:
        raise ValueError('Expect camera product line: "D435" or "L515" or "Kinect".')
    
    if cam.device_product_line == "L500":
        model.size = (540, 960)

    elif cam.device_product_line == "AzureKinect":
        model.size = (720, 1280)
    
    else:
        model.size = (480, 640)

    print("INFO: Start prediction with streaming mode...")

    alpha = 0.6
    
    with torch.no_grad():
        while True:

            cv2.namedWindow('rgb image')
            cv2.namedWindow('depth image')

            rgb_image, depth_image = cam.stream()
            
            depth_image = depth_image.astype(np.int16)
            depth_image = duplicate_channel_cv(depth_image, camera=cam.device_product_line)

            key = cv2.waitKey()
            if key == ord('p'):
                cv2.destroyAllWindows()
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
                rgb_image_t = array_to_tensor(rgb_image, dtype='float').unsqueeze(0).to(device=device)/255

                if cam.device_product_line == "L500":
                    depth_image_t = array_to_tensor(depth_image, dtype='float').unsqueeze(0).to(device=device)*0.00025
                else:
                    depth_image_t = array_to_tensor(depth_image, dtype='float').unsqueeze(0).to(device=device)*0.001

                t1 = time.time()
                out = model(rgb_image_t, depth_image_t)
                print("INFO: Prediction took {:.3f}s".format(time.time()-t1))

                prediction_map = process_prediction_map(out)
                suction_point, max_pred_val = get_suction_point(prediction_map)

                org_prediction_map, heatmap = prediction_to_heatmap(prediction_map, max_pred_val, cam.device_product_line)
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

                combined_heatmap = cv2.addWeighted(heatmap, alpha, rgb_image, 1-alpha, 0)
                
                try:
                    cv2.circle(combined_heatmap, (int(suction_point[1]), int(suction_point[0])), 3, (0, 250, 0), 2)
                except:
                    print("NO SUCTION POINT!")

                cv2.imshow("Prediction Result", combined_heatmap)
                cv2.waitKey(0)                

elif in_args.mode == 'dataset':
    print('INFO: start prediction with dataset mode...')

    if in_args.camera_model == 'L515' or in_args.example_modality == 'real':
        model.size = (540, 960)
        camera_model = 'L500'

    elif in_args.camera_model == 'Kinect':
        model.size = (720, 1280)
        camera_model = 'AzureKinect'

    else:
        camera_model = 'D400'

    # example images
    if in_args.example_modality == 'real':

        dataset_path = 'examples/real'
        rgb_data_list = glob.glob(dataset_path+'/rgb/*.png')
        depth_data_list = glob.glob(dataset_path+'/depth/*.npy')

    elif in_args.example_modality == 'sim':

        dataset_path = 'examples/sim'
        rgb_data_list = glob.glob(dataset_path+'/rgb/*.png')
        depth_data_list = glob.glob(dataset_path+'/depth/*.npy')

    else:
        raise ValueError('example_modality should be "real" or "sim"!')

    rgb_data_list.sort()
    depth_data_list.sort()

    print('number of test images:',len(rgb_data_list))
    with torch.no_grad():
        for rgb, depth in zip(rgb_data_list, depth_data_list):

            rgb_image = cv2.cvtColor(cv2.imread(rgb), cv2.COLOR_BGR2RGB)
            depth_image = np.load(depth)
            depth_image = duplicate_channel_cv(depth_image, 3, camera_model)
            rgb_image_t = array_to_tensor(rgb_image, dtype='float').unsqueeze(0).to(device=device)/255
            depth_image_t = array_to_tensor(depth_image, dtype='float').unsqueeze(0).to(device=device)
            print("INFO: Predicting suction points...")
            t1 = time.time()
            out = model(rgb_image_t, depth_image_t)
            print("INFO: Prediction took {:.3f}s".format(time.time()-t1))

            prediction_map = process_prediction_map(out)

            suction_point, max_pred_val = get_suction_point(prediction_map)

            org_prediction_map, heatmap = prediction_to_heatmap(prediction_map, max_pred_val, camera_model)
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            alpha = 0.6
            combined_heatmap = cv2.addWeighted(heatmap, alpha, rgb_image, 1-alpha, 0)
            
            try:
                cv2.circle(combined_heatmap, (int(suction_point[1]), int(suction_point[0])), 3, (0, 250, 0), 2)
            except:
                pass

            cv2.imshow("combined", combined_heatmap)
            cv2.waitKey(0)

else:
    raise ValueError('mode should be either "streaming" or "dataset".')