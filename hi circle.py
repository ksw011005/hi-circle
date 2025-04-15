import cv2 as cv
import numpy as np

board_pattern = (9, 6)
board_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

video = cv.VideoCapture("/Users/sungwoo/Desktop/chess.mp4")

board_cellsize = 0.025

circle_center = np.array([4.5, 3.0, -0.5]) * board_cellsize
circle_radius = 0.5 * board_cellsize

# 원 궤적 3D 포인트
circle_3d = []
for theta in np.linspace(0, 2*np.pi, 36):
    x = circle_center[0] + circle_radius * np.cos(theta)
    y = circle_center[1] + circle_radius * np.sin(theta)
    z = circle_center[2]
    circle_3d.append([x, y, z])
circle_3d = np.array(circle_3d, dtype=np.float32)

# 체스보드의 3D 포인트
obj_points = board_cellsize * np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])

K = np.array([[800, 0, 320],
              [0, 800, 240],
              [0, 0, 1]], dtype=np.float64)
dist_coeff = np.zeros((5, 1))

paused = False  # 일시정지 상태 변수

while True:
    if not paused:
        ret, img = video.read()
        if not ret:
            break

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        complete, img_points = cv.findChessboardCorners(gray, board_pattern, None)
        
        if complete:
            img_points = cv.cornerSubPix(gray, img_points, (11, 11), (-1, -1), board_criteria)

            ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)

            # 원 그리기
            projected_circle, _ = cv.projectPoints(circle_3d, rvec, tvec, K, dist_coeff)
            projected_circle = np.int32(projected_circle).reshape(-1, 1, 2)
            cv.polylines(img, [projected_circle], isClosed=True, color=(0, 255, 255), thickness=2)

            # 카메라 위치 출력
            R, _ = cv.Rodrigues(rvec)
            p = (-R.T @ tvec).flatten()
            info = f'XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]'
            cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

    cv.imshow("Pose Estimation with Circle", img)

    key = cv.waitKey(30) & 0xFF
    if key == 27:  # ESC
        break
    elif key == 32:  # Spacebar
        paused = not paused  # 일시정지 토글

video.release()
cv.destroyAllWindows()
