# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers

The process behind converting custom layers involves register custom layers to the Model Optimizer. Depending on the framework the process could be different. For TensorFlow which was the model that I used, after register the custom layer to the Model Optimizer it is necessary to replace the unsupported subgraph with a different subgraph.

Some of the potential reasons for handling custom layers are to avoid the problem of unsupported layers of a specific framework. This way, using custom layers helps in the conversion of the model in an IR avoiding the accuracy loss and speeding up the perfomance. 

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were using lantency and memory comparison in the models pre-converted ou post-converted.

The difference between model accuracy pre- and post-conversion was 1280 and 890 microsenconds which shows that the post-convesion has a faster performance.

The size of the model pre- and post-conversion was 562Mb and 281Mb which shows that OpenVino reduces a lot the model size helping to run the model on Edge.

In relation to accuracy, the pre-converted model has a slightly higher accuracy compared to the post-converted model. However, this small loss in accuracy in the post-converted moldel results in a faster model which it is pretty good to run a model at the Edge. So, this is a good trade off. However, with the probrability threshold of 0.35 the model acurracy was 100%. The number of frames a person is detected with bounding box was equal to the number of frames person was present for all the dectections. The pictures below show a case for the sixth perso that appears on video.



## Assess Model Use Cases

Some of the potential use cases of the people counter app are in scenarios where it is required to control the flow of people in a specific ambient. For example to limit the people in a bar following the firefighters code to avoid problems in cases of fire. In the actual pandamic scenario it can be very useful as well where retails have to control the people inside the rooms to avoid diseases contaminations. This cases the app could send and alert to the manager in case the number of people in a ambient is outreach.

Each of these use cases would be useful because the number of people would controlled automatically avoiding the use of people to perform the same task.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows:
The lighting makes the detection of a model easier once it can facilitate the image shape and color to be detected. The model accuracy helps getting a higher rate of right detections and avoinding wrong detections. The camera and image size can have and impact facilitating the image detection through a higher pixel quality.

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried  the following model:

- Model 1: [Faster_rcnn_inception_v2_coco_2018_01_28]
  - [https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md]
  - I converted the model to an Intermediate Representation with the following arguments...
    tar -xvf ssd_inception_v2_coco_2018_01_28.tar.gz
    cd ssd_inception_v2_coco_2018_01_28
    python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json
   - I tried to improve the model for the app by reducing the probability threshold to 0.35 and this way getting a better detection for this scenario.
  


