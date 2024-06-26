import cv2
import mediapipe as mp
import numpy as np
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

cap = cv2.VideoCapture(0)

# 이긴 횟수를 0으로 초기화시킴
leftHand_wins = 0
rightHand_wins = 0

# 이전에 승리한 시간을 기록
win_time = time.time()

# 승리 판정 제한 시간을 3초로 만듦
win_limit_sec = 3

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

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
                                cv2.imshow('Game', img)
                                cv2.waitKey(5000)    # 5초 후 게임을 종료
                                cap.release()
                                cv2.destroyAllWindows()
                                break
                        else:
                            rightHand_wins += 1
                            if rightHand_wins == 5:
                                cv2.putText(img, text='Finish', org=(int(img.shape[1] / 2), 50),
                                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=3)
                                cv2.putText(img, text='Winner Right', org=(int(img.shape[1] / 2), 100),
                                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 255, 0), thickness=3)
                                cv2.imshow('Game', img)
                                cv2.waitKey(5000)  # 5초 후 게임을 종료
                                cap.release()
                                cv2.destroyAllWindows()
                                break
                        win_time = current_time

                # 왼쪽 손의 이긴 횟수를 표시
                cv2.putText(img, text=str(leftHand_wins), org=(50, img.shape[0] - 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=2, color=(0, 255, 255), thickness=3)
                # 오른쪽 손의 이긴 횟수를 표시
                cv2.putText(img, text=str(rightHand_wins), org=(img.shape[1] - 100, img.shape[0] - 50),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 255, 255), thickness=3)

                cv2.putText(img, text=text, org=(int(img.shape[1] / 2), 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=2, color=(0, 0, 255), thickness=3)

    cv2.imshow('Game', img)

    if cv2.waitKey(1) == ord('q'):
        break
    