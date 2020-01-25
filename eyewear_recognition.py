from imutils import face_utils
from face_alignment import *
from helpers import *


p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)
fa = FaceAligner(predictor, desiredFaceWidth=256)
face_cascade=cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
counter=1
measure_list=[]
recognition_result="None"
color=(0,0,0)

# frame_width,frame_height=640,480
# out = cv2.VideoWriter("./result/result.avi",cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))



while True:
    _, img = cap.read()
    global_result=img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for x,y,w,h in faces:

        roi_color = img[y:y + h, x:x + w] #crop face from image
        rect = cs_to_rect([x, y, w, h])
        # face alignment is used so that angles do no affect calculation
        img = fa.align(img, gray, rect)
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        #finding edges around face
        edgeness = cv2.Canny(gray, 50, 100)

        x1 = shape[18][0]
        x2 = shape[25][0]
        y1 = shape[18][1]
        y2 = shape[30][1]

        edgeness = edgeness[y1:y2, x1:x2]

        #filter for nose removing some edges around nose
        lower_nose_x = shape[31][0]-x1
        upper_nose_x = shape[35][0]-x1
        lower_nose_y = shape[27][1]-y1
        upper_nose_y = y2-y1

        #updating edgeness
        try:
            for edge_x in range(lower_nose_x, upper_nose_x):
                for edge_y in range(lower_nose_y, upper_nose_y):
                    edgeness[edge_y][edge_x] = 0
        except Exception as e:
            pass
        """
        lower_left_eye_x = shape[18][0] - x1
        upper_left_eye_x = shape[21][0] - x1
        lower_left_eye_y = shape[18][1] - y1
        upper_left_eye_y = shape[41][1] - y1
        """
        try:
            for left_eye_x in range(shape[18][0] - x1, shape[21][0] - x1):
                for left_eye_y in range(shape[18][1] - y1, shape[41][1] - y1):
                    edgeness[left_eye_y][left_eye_x] = 0
        except Exception as e:
            pass
        """
        lower_right_eye_x = shape[22][0] - x1
        upper_right_eye_x = shape[25][0] - x1
        lower_right_eye_y = shape[25][1] - y1
        upper_right_eye_y = shape[46][1] - y1
        """
        try:
            for right_eye_x in range(shape[22][0] - x1, shape[25][0] - x1):
                for right_eye_y in range(shape[25][1] - y1, shape[46][1] - y1):
                    edgeness[right_eye_y][right_eye_x] = 0
        except Exception as e:
            pass

        #measure edgness and normalzing to get better result
        try:
            if counter < 6:
                measure1 = sum(sum(edgeness / 255)) / (np.shape(edgeness)[0] * np.shape(edgeness)[1])*100
                measure_list.append(measure1)
                counter=counter+1
        except Exception as e:
            pass
        print(measure_list)
        if len(measure_list) == 5:
            glass_list= []
            for num in measure_list:
                if num > 2:
                    glass_list.append('glass')
            measure_list=[]
            counter = 1

            if len(glass_list) >= 4 :
                recognition_result="with glass"
                color=(0, 255, 0)

            else:
                recognition_result = "without glass"
                color=(0, 0, 255)

    cv2.putText(global_result, recognition_result, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
                    cv2.LINE_AA)
    cv2.putText(global_result, "press q to exit", (350, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2,
                cv2.LINE_AA)
    cv2.imshow('glass wear detection',global_result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()