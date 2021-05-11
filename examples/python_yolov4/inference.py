import migraphx
import numpy as np
import cv2
from matplotlib.pyplot import imshow
from PIL import Image
import time
import image_processing as ip

input_size = 416

original_image = cv2.imread("test1.jpg")
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
original_image_size = original_image.shape[:2]

image_data = ip.image_preprocess(np.copy(original_image), [input_size, input_size])
image_data = image_data[np.newaxis, ...].astype(np.float32)

model = migraphx.load("../python_model_optimizer/saved_models/yolov4_fp16.msgpack", format="msgpack")
#model.print()
input_map = model.get_parameter_shapes()
input_name = next(iter(input_map))
input_argument = migraphx.argument(image_data)

start = time.time()
outputs = model.run({input_name: input_argument})
stop = time.time()
print("Eval time: ", stop-start)

detections = [np.ndarray(shape=out.get_shape().lens(), buffer=np.array(out.tolist()), dtype=float) for out in outputs]

ANCHORS = "./yolov4_anchors.txt"
STRIDES = [8, 16, 32]
XYSCALE = [1.2, 1.1, 1.05]

ANCHORS = ip.get_anchors(ANCHORS)
STRIDES = np.array(STRIDES)

pred_bbox = ip.postprocess_bbbox(detections, ANCHORS, STRIDES, XYSCALE)
bboxes = ip.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.25)
bboxes = ip.nms(bboxes, 0.213, method='nms')
image = ip.draw_bbox(original_image, bboxes)

image = Image.fromarray(image)
image.show()
