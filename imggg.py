import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone

model = YOLO('best (1).pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        point = [x, y]
        print(point)
        print('param',param)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

#class ImageCapture:
 #   def __init__(self, image_paths):
  #      self.image_paths = image_paths
   #     self.current_index = 0

    #def read(self):
     #   if self.current_index < len(self.image_paths):
      #      image_path = self.image_paths[self.current_index]
       #     frame = cv2.imread(image_path)
        #    self.current_index += 1
         #   return True, frame
        #else:
           # return False, None

# Example usage:
#image_paths = [r"C:\Users\madha\Downloads\Personal Protection Detetction.v1i.yolov8-obb\Video1_197_jpg.rf.60ce2752c8cbd823e90f81fc25776067.jpg"]
#image_paths = [r"C:\Users\muham\Desktop\PPE\Video1_197_jpg.rf.60ce2752c8cbd823e90f81fc25776067.jpg"]
#image_capture = ImageCapture(image_paths)
video_capture=cv2.VideoCapture(0)
my_file = open("coco1.txt", "r")
data = my_file.read()
class_list = data.split("\n") 

while True:
    #ret, frame = image_capture.read()
    ret, frame = video_capture.read()
    if not ret:
        break

#     cv2.imshow('RGB', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# 
#     count = 0

    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    a = results[0].boxes.data
    print('a',a)
    px = pd.DataFrame(a).astype("float")
    
    ppe_found = False 
   
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'Goggles' in c:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
            ppe_found = True

        elif 'boots' in c:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
            ppe_found = True

        elif 'gloves'in c:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
            ppe_found = True
        else:
            print('Warning: NO PPE KIT FOUND')
            image =cv2.imread("warning.png")
            cv2.imshow("alert",image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    if not ppe_found:
        print('Warning: NO PPE KIT FOUND')
        image =cv2.imread("warning.png")
        cv2.imshow("alert",image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
video_capture.release()
cv2.destroyAllWindows()
