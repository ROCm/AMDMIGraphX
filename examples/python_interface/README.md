## Python Interface
The Python interface has been simplified so that it is no longer necessary to copy in parameters.  An updated version of the webcam example from <a href="https://github.com/ROCmSoftwarePlatform/AMDMIGraphX/wiki/Getting-started:-using-the-new-features-of-MIGraphX-0.2">MIGraphX examples v0.2</a> is shown below.

A careful comparison with previous example shows it is no longer necessary to allocate the parameters on the GPU or to explicitly copy parameters to the GPU or results from the GPU.  The Python API sets offload_copy=True by default when compiling the model (see [migraphx.compile](https://rocmsoftwareplatform.github.io/AMDMIGraphX/doc/html/reference/py.html#migraphx.compile)) which will insert copies into the migraphx::program.
```
import numpy as np
import cv2
import json
import migraphx

# video settings
cap = cv2.VideoCapture(0)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 240)
ret, frame = cap.read()

# neural network settings
model = migraphx.parse_onnx("resnet50.onnx")
model.compile(migraphx.get_target("gpu"))

# get labels
with open('imagenet_class_index.json') as json_data:
    class_idx = json.load(json_data)
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

# primary loop to read webcam images
count = 0
while (True):
    # capture frame by frame
    ret, frame = cap.read()

    if ret: # check - some webcams need warmup operations on the frame
        cropped = frame[16:304,8:232]    # 224x224

        trans = cropped.transpose(2,0,1) # convert HWC to CHW

        # convert to float, normalize and make batch size = 1
        image = np.ascontiguousarray(
            np.expand_dims(trans.astype('float32')/256.0,0))

        # display the frame
        cv2.imshow('frame',cropped)

        migraphx_result = model.run({'0':image})

        result = np.array(migraphx_result,copy=False)

        idx = np.argmax(result[0])

        print(idx2label[idx], " ", result[0][idx])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# when everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
```
