"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
   
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    
    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device, args.cpu_extension)
    network_input_shape = infer_network.get_input_shape()
    
    ### TODO: Handle the input stream ###
    # Checks for live feed
    if args.input == 'CAM':
        input_stream = 0
        single_image_mode = False

    # Checks for input image
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') :
        single_image_mode = True
        input_stream = args.input

    # Checks for video file
    else:
        single_image_mode = False
        input_stream = args.input
        assert os.path.isfile(args.input), "file doesn't exist"
    
    cap = cv2.VideoCapture(input_stream)
    cap.open(input_stream)
    width = int(cap.get(3))
    height = int(cap.get(4))
    
    input_shape = network_input_shape['image_tensor']  
    # Iniatilize variables
    current_count=0
    total_count=0
    init_time=0
    exit_time=0
    request_id=0
    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        
        ### TODO: Pre-process the image as needed ###
        p_frame = cv2.resize(frame, (input_shape[3], input_shape[2]))
        p_frame = p_frame.transpose((2, 0, 1))
        p_frame = p_frame.reshape(1, *p_frame.shape) 
        
        ### TODO: Start asynchronous inference for specified request ###
        network_input = {'image_tensor': p_frame, 'image_info': p_frame.shape[1:]}
        infer_network.exec_net(request_id, network_input)
        
        ### TODO: Wait for the result ###
        if infer_network.wait() == 0: 
            
            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output()
            
            ### TODO: Extract any desired stats from the results ###
            people_detected_frame = 0
            prob_confidence = result[0, 0, :, 2]
            for i, prob in enumerate(prob_confidence):
                if prob > prob_threshold:
                    people_detected_frame += 1
                    box = result[0, 0, i, 3:]
                    xmin = int(box[0] * width)
                    ymin = int(box[1] * height)
                    xmax = int(box[2] * width)
                    ymax = int(box[3] * height)
                    frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
            
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            #Detecting when people appear in the scene
            if people_detected_frame > current_count:
                init_time = time.time()
                duration_off = int(init_time - exit_time)
                current_count = people_detected_frame
                if duration_off > 18:
                    total_count += people_detected_frame   
                    client.publish("person", json.dumps({"total": total_count}))
                 
            #Detecting when people leave the scene
            if people_detected_frame < current_count:
                exit_time = time.time()
                duration_on = int(exit_time - init_time)
                current_count = people_detected_frame
                # Publish messages to the MQTT server
                if duration_on > 18:
                    client.publish("person/duration", json.dumps({"duration": duration_on}))
            
            client.publish("person", json.dumps({"count": current_count}))
            
                        
        ### TODO: Send the frame to the FFMPEG server ###
        
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
        ### TODO: Write an output image if `single_image_mode` ###
        if single_image_mode:
            cv2.imwrite('output_image.jpg', frame)

    cap.release()
    cv2.destroyAllWindows()

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()