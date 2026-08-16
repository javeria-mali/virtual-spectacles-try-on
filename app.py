import cv2
import mediapipe as mp
import math
import os


# ==========================================
# 1. MediaPipe Face Mesh
# ==========================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ==========================================
# 2. Load Glasses
# ==========================================

glasses_list = {
    "1": cv2.imread(
        "models/glasses.png",
        cv2.IMREAD_UNCHANGED
    ),

    "2": cv2.imread(
        "models/glasses_black.png",
        cv2.IMREAD_UNCHANGED
    ),

    "3": cv2.imread(
        "models/glasses_round.png",
        cv2.IMREAD_UNCHANGED
    )
}


# ==========================================
# 3. Check Images
# ==========================================

for key, image in glasses_list.items():

    if image is None:
        print(f"WARNING: Glasses {key} ki image nahi mili!")


# Default glasses
current_glasses = glasses_list["1"]

print("Glasses loaded successfully!")


# ==========================================
# 4. Create Outputs Folder
# ==========================================

os.makedirs("outputs", exist_ok=True)


# ==========================================
# 5. Rotate Image
# ==========================================

def rotate_image(image, angle):

    h, w = image.shape[:2]

    center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    rotated = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    return rotated


# ==========================================
# 6. Overlay Transparent Glasses
# ==========================================

def overlay_glasses(frame, overlay, x, y):

    h, w = overlay.shape[:2]

    frame_h, frame_w = frame.shape[:2]

    # Left boundary
    if x < 0:

        overlay = overlay[:, -x:]

        w = overlay.shape[1]

        x = 0

    # Top boundary
    if y < 0:

        overlay = overlay[-y:, :]

        h = overlay.shape[0]

        y = 0

    # Right boundary
    if x + w > frame_w:

        overlay = overlay[:, :frame_w - x]

        w = overlay.shape[1]

    # Bottom boundary
    if y + h > frame_h:

        overlay = overlay[:frame_h - y, :]

        h = overlay.shape[0]

    if w <= 0 or h <= 0:
        return

    # Transparent PNG
    if overlay.shape[2] == 4:

        alpha = overlay[:, :, 3] / 255.0

        for c in range(3):

            frame[
                y:y + h,
                x:x + w,
                c
            ] = (
                alpha * overlay[:, :, c]
                +
                (1 - alpha)
                * frame[
                    y:y + h,
                    x:x + w,
                    c
                ]
            )

    else:

        frame[
            y:y + h,
            x:x + w
        ] = overlay[:, :, :3]


# ==========================================
# 7. Open Camera
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Camera open nahi ho rahi!")

    exit()


print()
print("================================")
print("   VIRTUAL TRY-ON SPECTACLES")
print("================================")
print()
print("1 = White Glasses")
print("2 = Black Glasses")
print("3 = Round Glasses")
print("S = Take Photo")
print("Q = Quit")
print()
print("Camera Started!")


# ==========================================
# 8. Main Loop
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("Camera se frame nahi mil raha!")

        break


    # Mirror camera
    frame = cv2.flip(frame, 1)

    frame_height, frame_width = frame.shape[:2]


    # ======================================
    # Face Detection
    # ======================================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = face_mesh.process(rgb_frame)


    # ======================================
    # Face Found
    # ======================================

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            # ------------------------------
            # Eye Landmarks
            # ------------------------------

            left_eye = face_landmarks.landmark[33]

            right_eye = face_landmarks.landmark[263]


            left_x = int(
                left_eye.x * frame_width
            )

            left_y = int(
                left_eye.y * frame_height
            )


            right_x = int(
                right_eye.x * frame_width
            )

            right_y = int(
                right_eye.y * frame_height
            )


            # ------------------------------
            # Distance Between Eyes
            # ------------------------------

            eye_distance = math.sqrt(
                (right_x - left_x) ** 2
                +
                (right_y - left_y) ** 2
            )


            # ------------------------------
            # Glasses Width
            # ------------------------------

            glasses_width = int(
                eye_distance * 2.20
            )


            # ------------------------------
            # Original Image Size
            # ------------------------------

            original_height, original_width = (
                current_glasses.shape[:2]
            )


            # ------------------------------
            # Glasses Height
            # ------------------------------

            glasses_height = int(
                glasses_width
                * original_height
                / original_width
            )


            # ------------------------------
            # Resize
            # ------------------------------

            resized_glasses = cv2.resize(
                current_glasses,
                (
                    glasses_width,
                    glasses_height
                ),
                interpolation=cv2.INTER_AREA
            )


            # ------------------------------
            # Head Tilt
            # ------------------------------

            angle = math.degrees(
                math.atan2(
                    right_y - left_y,
                    right_x - left_x
                )
            )


            # ------------------------------
            # Rotate
            # ------------------------------

            rotated_glasses = rotate_image(
                resized_glasses,
                -angle
            )


            # ------------------------------
            # Center Between Eyes
            # ------------------------------

            center_x = int(
                (left_x + right_x) / 2
            )

            center_y = int(
                (left_y + right_y) / 2
            )


            # ------------------------------
            # Move Glasses Down
            # ------------------------------

            center_y += int(
                glasses_height * 0.03
            )


            # ------------------------------
            # Position
            # ------------------------------

            rotated_height, rotated_width = (
                rotated_glasses.shape[:2]
            )

            x = int(
                center_x - rotated_width / 2
            )

            y = int(
                center_y - rotated_height / 2
            )


            # ------------------------------
            # Overlay
            # ------------------------------

            overlay_glasses(
                frame,
                rotated_glasses,
                x,
                y
            )


            # ------------------------------
            # Status
            # ------------------------------

            cv2.putText(
                frame,
                "Virtual Try-On",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )


    # ======================================
    # Face Not Found
    # ======================================

    else:

        cv2.putText(
            frame,
            "Face Not Detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    # ======================================
    # Show Glasses Options
    # ======================================

    cv2.putText(
        frame,
        "1: White",
        (20, frame_height - 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "2: Black",
        (20, frame_height - 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "3: Round",
        (20, frame_height - 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "S: Take Photo",
        (20, frame_height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # ======================================
    # Show Camera
    # ======================================

    cv2.imshow(
        "Try-On Spectacles",
        frame
    )


    # ======================================
    # Keyboard Controls
    # ======================================

    key = cv2.waitKey(1) & 0xFF


    # White glasses
    if key == ord("1"):

        if glasses_list["1"] is not None:

            current_glasses = glasses_list["1"]

            print("White glasses selected!")


    # Black glasses
    elif key == ord("2"):

        if glasses_list["2"] is not None:

            current_glasses = glasses_list["2"]

            print("Black glasses selected!")


    # Round glasses
    elif key == ord("3"):

        if glasses_list["3"] is not None:

            current_glasses = glasses_list["3"]

            print("Round glasses selected!")


    # Take Photo
    elif key == ord("s"):

        filename = "outputs/tryon_photo.jpg"

        success = cv2.imwrite(
            filename,
            frame
        )

        if success:

            print()
            print("Photo successfully saved!")
            print("Location:", filename)
            print()

        else:

            print("Photo save nahi ho saki!")


    # Quit
    elif key == ord("q"):

        break


# ==========================================
# 9. Close Everything
# ==========================================

cap.release()

cv2.destroyAllWindows()

face_mesh.close()

print("Virtual Try-On Closed!")