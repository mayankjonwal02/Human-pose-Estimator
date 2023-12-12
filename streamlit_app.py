import streamlit as st 
import numpy as np 
import cv2
import tensorflow as tf
import mediapipe as mp
from pose_labels_dict import PoseData


model = tf.keras.models.load_model("pose_model_2")
cap = 0
st.title('Human Pose Estimator')

button_clicked = st.empty()
mywords = st.empty()

# button_clicked.button("Start Recognition", key="my_button")
# button_clicked1 = st.empty()
# button_clicked1.button("**Click me!*1*", key="stop_button")

def execute():
    sentence = []
    previousword = ""
    hand_detector = mp.solutions.hands.Hands(static_image_mode=False)
    drawing_utils = mp.solutions.drawing_utils
    face_detector = mp.solutions.face_mesh.FaceMesh(static_image_mode=False)
    pose_detector = mp.solutions.pose.Pose(static_image_mode=False)
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 0, 255)  # White color in BGR
    text = "predicting..."
    font_scale = 1
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    pose_data = PoseData()
    labels = pose_data.fetch_labels()

    cap = cv2.VideoCapture(0)
    video = []
    text = "Predicting..."
    while True :

        _,frame = cap.read()
        frame = cv2.flip(frame , 1)
        rgb_frame = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
        hand_output = hand_detector.process(rgb_frame)
        hands = hand_output.multi_hand_landmarks
        face_output = face_detector.process(rgb_frame)
        faces = face_output.multi_face_landmarks
        pose_output = pose_detector.process(rgb_frame)
        poses = pose_output.pose_landmarks
        fheight , fwidth , _ = frame.shape

        if poses:
            drawing_utils.draw_landmarks(frame , poses , mp.solutions.pose.POSE_CONNECTIONS , landmark_drawing_spec=drawing_utils.DrawingSpec(color=(255,255,255),
                                                                                thickness=3, circle_radius=3),
                                    connection_drawing_spec=drawing_utils.DrawingSpec(color=(49,125,237),
                                                                                thickness=2, circle_radius=2))

        if faces:
            for face in faces:
                drawing_utils.draw_landmarks(frame , face , mp.solutions.face_mesh.FACEMESH_TESSELATION , None , mp.solutions.drawing_styles.DrawingSpec(color=(0,0,255),thickness=1,circle_radius=1))
                drawing_utils.draw_landmarks(frame , face , mp.solutions.face_mesh.FACEMESH_CONTOURS , None , mp.solutions.drawing_styles.DrawingSpec(color=(0,0,255),thickness=2,circle_radius=1))
                # drawing_utils.draw_landmarks(frame , face , mp.solutions.face_mesh.FACEMESH_IRISES , None , mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style())
        
        if hands :
            hand = hands[0]
            # drawing_utils.draw_landmarks(frame , hand , mp.solutions.hands.HAND_CONNECTIONS )
            videofeatures = []
            landmarks = hand.landmark
            
            for id , landmark in enumerate(landmarks):
                # x = int(landmark.x*fwidth)
                # y = int(landmark.y*fheight)
                # cv2.circle(frame , (x,y) , 5 , (255,0,0) , 3)
                videofeatures.append(landmark.x)
                videofeatures.append(landmark.y)

            for hand in hands:
                drawing_utils.draw_landmarks(frame , hand , mp.solutions.hands.HAND_CONNECTIONS )
                for id , landmark in enumerate(hand.landmark):
                    x = int(landmark.x*fwidth)
                    y = int(landmark.y*fheight)
                    cv2.circle(frame , (x,y) , 5 , (255,0,0) , 3)
            
            video.append(videofeatures)
            print(len(video))
            
            if(len(video) == 15):
                input_data = np.array([video])
                # print(input_data)
                # input_data = input_data.reshape((input_data.shape[0], 1, input_data.shape[1]))
                prediction = model.predict(input_data)
                # Get the predicted class index (assuming it's a classification task)
                predicted_class = np.argmax(prediction)

                print(f"Predicted class index: {predicted_class}")
                text = labels[predicted_class]
                video.remove(video[0])
        else:
            text = "Predicting..."
                
                
        position = (fwidth - text_size[0] - 10, 30)
        cv2.putText(frame, str(text), position, font, font_scale, color, thickness, cv2.LINE_AA)
        # cv2.imshow("myhands",frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        button_clicked.image(frame)
        if previousword != text and text != "Predicting..." and text != "Random":
            previousword = text 
            if(len(sentence) == 10):
                sentence.remove(sentence[0])
            sentence.append(previousword)
        phrase = " ".join(sentence)
        mywords.text(phrase)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            
            break
    cap.release()
    cv2.destroyAllWindows()


execute()
# if button_clicked:
#     execute()


# if button_clicked1:
#     cap.release()
#     cv2.destroyAllWindows()

