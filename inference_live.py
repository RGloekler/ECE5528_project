# Ryan Gloekler
# ECE 528 - Dec. 10 2021
# Object detection and Inference with openCV

from tensorflow.keras.preprocessing import image
import os, cv2, time, sys
import numpy as np
import tensorflow as tf

# initialize global variables for main loop
#img_classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
img_classes = ['can', 'glass', 'plastic']
# font variables
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
color = (255, 0, 0)
thickness = 2


def infer(interpreter):
  if not os.path.exists('data/data.png'): return

  # load an image and preprocess/resize it
  test_img = image.load_img('data/data.png', target_size=(256, 256))

  # perform inference on that image, update prediction...
  x = image.img_to_array(test_img)
  x = np.expand_dims(x, axis=0)

  # run inference
  interpreter.set_tensor(input_details[0]['index'], x)
  interpreter.invoke()

  output_data = interpreter.get_tensor(output_details[0]['index'])
  prediction = np.argmax(output_data[0])
  pred_str = img_classes[prediction]

  print(pred_str)

  # set the percent confidence in the prediction
  confidence = max(output_data[0]) / sum(output_data[0]) * 100
  confidence = round(confidence, 2)

  return pred_str, confidence

# main execution loop
if __name__ == "__main__":
  # initialize the camera (0 for local webcam), run 'lsusb' for other devices
  cam = cv2.VideoCapture(2)
  cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
  cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

  # Load the TFLite model and allocate tensors.
  interpreter = tf.lite.Interpreter(model_path="models/model_kaggleds_res50_80_10.tflite")
  interpreter.allocate_tensors()

  # Get input and output tensors.
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # variables used in main inference loop
  frame_counter = 0
  val_prediction = 'NONE'
  confidence = 0.0

  while True:
    ret_val, img = cam.read()

    # overlay text on the image
    image_text = cv2.putText(img, val_prediction + ' ' + str(confidence) + '%', org, font,
    fontScale, color, thickness, cv2.LINE_AA)

    cv2.imshow('Live Inference', image_text)

    if frame_counter % 10 == 0:
        ret_val, img = cam.read()
        cv2.imwrite('data/data.png', img)

    if frame_counter % 15 == 0:
        tic = time.perf_counter()
        val_prediction, confidence = infer(interpreter)
        toc = time.perf_counter()

        inf_time = toc - tic

        # if timing flag is set, write inference time
        if len(sys.argv) >= 2 and sys.argv[1] == '-timing':
            timing_file = open("data/timer_data.txt", "a")
            timing_file.write(str(inf_time * 1000) + '\n') # write in ms

    if cv2.waitKey(1) == 27:
        exit(0)
        break  # esc to quit

    frame_counter += 1

  cam.release()
  cv2.destroyAllWindows()
