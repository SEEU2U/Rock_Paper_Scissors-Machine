import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import mediapipe as mp
import numpy as np
import os
import time

max_num_hands = 2
gesture = {
    0: 'fist', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
    6: 'six', 7: 'rock', 8: 'spiderman', 9: 'yeah', 10: 'ok',
}
rps_gesture = {0: 'rock', 5: 'paper', 9: 'scissors'}

# MediaPipe 손 모델
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# 제스쳐 인식 모델
file = np.genfromtxt('data/gesture_train.csv', delimiter=',')
angle = file[:, :-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

# 전역 변수 초기화
leftHand_wins = 0
rightHand_wins = 0

# 이전에 승리한 시간을 기록
win_time = time.time()

# 승리 판정 제한 시간을 3초로 만듦
win_limit_sec = 3

def start_game(root, canvas, webcam_label, start_button):
    global leftHand_wins, rightHand_wins, win_time  # 전역 변수를 선언합니다.

    cap = cv2.VideoCapture(0)

    def open_main_window():
        cap.release()
        root.destroy()
        main()

    def show_frame():
        global leftHand_wins, rightHand_wins, win_time  # 전역 변수를 선언합니다.
        ret, img = cap.read()
        if not ret:
            return

        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = hands.process(img)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks is not None:
            rps_result = []

            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 3))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]

                # 조인트의 각도 계산
                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]  # 부모 조인트
                v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]  # 자식 조인트
                v = v2 - v1  # [20,3]
                # Normalize v 정규화
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # 점의 각도계산
                angle = np.arccos(np.einsum('nt,nt->n',
                                             v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                             v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

                angle = np.degrees(angle)  # radian을 degree로 변환시킴

                # 제스쳐 추론
                data = np.array([angle], dtype=np.float32)
                ret, results, neighbours, dist = knn.findNearest(data, 3)
                idx = int(results[0][0])

                # 제스쳐의 결과를 그림
                if idx in rps_gesture.keys():
                    org = (int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0]))
                    cv2.putText(img, text=rps_gesture[idx].upper(), org=(org[0], org[1] + 20),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

                    rps_result.append({
                        'rps': rps_gesture[idx],
                        'org': org
                    })

                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                # 누가 이겼는지 확인
                if len(rps_result) >= 2:
                    winner = None
                    text = ''

                    if rps_result[0]['rps'] == 'rock':
                        if rps_result[1]['rps'] == 'rock': text = 'Tie'
                        elif rps_result[1]['rps'] == 'paper': text = 'Paper wins'; winner = 1
                        elif rps_result[1]['rps'] == 'scissors': text = 'Rock wins'; winner = 0
                    elif rps_result[0]['rps'] == 'paper':
                        if rps_result[1]['rps'] == 'rock': text = 'Paper wins'; winner = 0
                        elif rps_result[1]['rps'] == 'paper': text = 'Tie'
                        elif rps_result[1]['rps'] == 'scissors': text = 'Scissors wins'; winner = 1
                    elif rps_result[0]['rps'] == 'scissors':
                        if rps_result[1]['rps'] == 'rock': text = 'Rock wins'; winner = 1
                        elif rps_result[1]['rps'] == 'paper': text = 'Scissors wins'; winner = 0
                        elif rps_result[1]['rps'] == 'scissors': text = 'Tie'

                    if winner is not None:
                        current_time = time.time()
                        if current_time - win_time >= win_limit_sec:
                            cv2.putText(img, text='Winner', org=(rps_result[winner]['org'][0], rps_result[winner]['org'][1] + 70),
                                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 255, 0), thickness=3)

                            # 승리 제한 시간 이후 승리한 경우에 이긴 횟수 증가
                            if rps_result[winner]['org'][0] < img.shape[1] // 2:
                                leftHand_wins += 1
                                if leftHand_wins == 5:
                                    cv2.putText(img, text='Finish', org=(int(img.shape[1] / 2), 50),
                                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=3)
                                    cv2.putText(img, text='Winner Left', org=(int(img.shape[1] / 2), 100),
                                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 255, 0), thickness=3)
                                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                    img = Image.fromarray(img)
                                    imgtk = ImageTk.PhotoImage(image=img)
                                    webcam_label.imgtk = imgtk
                                    webcam_label.configure(image=imgtk)
                                    webcam_label.after(5000, open_main_window)
                                    return
                            else:
                                rightHand_wins += 1
                                if rightHand_wins == 5:
                                    cv2.putText(img, text='Finish', org=(int(img.shape[1] / 2), 50),
                                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=3)
                                    cv2.putText(img, text='Winner Right', org=(int(img.shape[1] / 2), 100),
                                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 255, 0), thickness=3)
                                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                    img = Image.fromarray(img)
                                    imgtk = ImageTk.PhotoImage(image=img)
                                    webcam_label.imgtk = imgtk
                                    webcam_label.configure(image=imgtk)
                                    webcam_label.after(5000, open_main_window)
                                    return
                            win_time = current_time

                    # 왼쪽 손의 이긴 횟수를 표시
                    cv2.putText(img, text=str(leftHand_wins), org=(50, img.shape[0] - 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=2, color=(0, 255, 255), thickness=3)
                    # 오른쪽 손의 이긴 횟수를 표시
                    cv2.putText(img, text=str(rightHand_wins), org=(img.shape[1] - 100, img.shape[0] - 50),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 255, 255), thickness=3)

                    cv2.putText(img, text=text, org=(int(img.shape[1] / 2), 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=2, color=(0, 0, 255), thickness=3)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        webcam_label.imgtk = imgtk
        webcam_label.configure(image=imgtk)

        root.after(10, show_frame)

    start_button.destroy()  # START 버튼을 제거합니다
    webcam_label.place(x=0, y=0, width=800, height=600)  # 웹캠 피드를 캔버스를 꽉 채우도록 배치합니다
    root.after(10, show_frame)
    main_button = tk.Button(root, text="MAIN", font=("Arial", 18), command=open_main_window)
    main_button.place(x=720, y=20)

def main():
    root = tk.Tk()
    root.title("Gesture Game")
    root.geometry("800x600")

    # 현재 스크립트의 디렉토리를 기준으로 이미지 경로 설정
    base_path = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(base_path, "image/1z.jpg")

    # Load the background image
    background_image = Image.open(image_path)
    background_image = background_image.resize((800, 600), Image.LANCZOS)
    background_photo = ImageTk.PhotoImage(background_image)

    # Create a canvas to display the background image
    canvas = tk.Canvas(root, width=800, height=600)
    canvas.pack(fill="both", expand=True)
    canvas.create_image(0, 0, image=background_photo, anchor="nw")

    # Add label and start button on top of the background image
    label = tk.Label(root, text="Welcome to the Gesture Game", font=("Arial", 24), bg="lightblue")
    label_window = canvas.create_window(400, 200, window=label)

    # Create a label for webcam feed
    webcam_label = tk.Label(root)

    start_button = tk.Button(root, text="Start", font=("Arial", 18), command=lambda: start_game(root, canvas, webcam_label, start_button))
    start_button_window = canvas.create_window(400, 300, window=start_button)

    root.mainloop()

if __name__ == "__main__":
    main()
