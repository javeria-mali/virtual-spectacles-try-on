import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import mediapipe as mp
import math
import av


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Virtual Spectacles Try-On",
    page_icon="👓",
    layout="wide"
)


# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #f8f9ff, #eef3ff);
}

.main-title {
    text-align: center;
    font-size: 45px;
    font-weight: 800;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #666;
    font-size: 18px;
    margin-bottom: 30px;
}

.card {
    background: white;
    padding: 22px;
    border-radius: 18px;
    box-shadow: 0 5px 20px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.info-box {
    background: #ffffff;
    padding: 15px;
    border-radius: 12px;
    border-left: 5px solid #6c63ff;
    margin-top: 15px;
}

.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 45px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">👓 Virtual Spectacles Try-On</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Try different spectacles virtually using your webcam ✨'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# GLASSES
# =========================================================

GLASSES_FILES = {
    "🤍 White Glasses": "models/glasses.png",
    "🖤 Black Glasses": "models/glasses_black.png",
    "⚫ Round Glasses": "models/glasses_round.png"
}


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("👓 Spectacles")

selected_glasses = st.sidebar.radio(
    "Choose your glasses:",
    list(GLASSES_FILES.keys())
)

st.sidebar.markdown("---")

st.sidebar.subheader("⚙️ Adjustments")

size_factor = st.sidebar.slider(
    "Glasses Size",
    min_value=1.80,
    max_value=2.60,
    value=2.20,
    step=0.05
)

vertical_offset = st.sidebar.slider(
    "Vertical Position",
    min_value=-0.15,
    max_value=0.15,
    value=0.03,
    step=0.01
)

st.sidebar.markdown("---")

st.sidebar.info(
    "📷 Click START below and allow camera permission."
)

st.sidebar.info(
    "👀 Keep your face clearly visible to the camera."
)


# =========================================================
# LOAD GLASSES IMAGE
# =========================================================

glasses_path = GLASSES_FILES[selected_glasses]

glasses_image = cv2.imread(
    glasses_path,
    cv2.IMREAD_UNCHANGED
)

if glasses_image is None:

    st.error(
        f"❌ Could not find glasses image:\n\n{glasses_path}"
    )

    st.stop()


# =========================================================
# MAKE SURE PNG HAS ALPHA
# =========================================================

if len(glasses_image.shape) == 2:

    glasses_image = cv2.cvtColor(
        glasses_image,
        cv2.COLOR_GRAY2BGRA
    )

elif glasses_image.shape[2] == 3:

    b, g, r = cv2.split(glasses_image)

    alpha = (
        cv2.cvtColor(
            glasses_image,
            cv2.COLOR_BGR2GRAY
        )
    )

    _, alpha = cv2.threshold(
        alpha,
        5,
        255,
        cv2.THRESH_BINARY
    )

    glasses_image = cv2.merge(
        [b, g, r, alpha]
    )


# =========================================================
# ROTATE IMAGE
# =========================================================

def rotate_image(image, angle):

    height, width = image.shape[:2]

    center = (
        width // 2,
        height // 2
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    rotated = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    return rotated


# =========================================================
# OVERLAY TRANSPARENT GLASSES
# =========================================================

def overlay_glasses(
    frame,
    overlay,
    x,
    y
):

    frame_height, frame_width = frame.shape[:2]

    overlay_height, overlay_width = overlay.shape[:2]


    # ---------------------------------------------
    # Completely outside frame
    # ---------------------------------------------

    if (
        x >= frame_width
        or y >= frame_height
        or x + overlay_width <= 0
        or y + overlay_height <= 0
    ):
        return


    # ---------------------------------------------
    # Crop left
    # ---------------------------------------------

    if x < 0:

        overlay = overlay[:, -x:]

        overlay_width = overlay.shape[1]

        x = 0


    # ---------------------------------------------
    # Crop top
    # ---------------------------------------------

    if y < 0:

        overlay = overlay[-y:, :]

        overlay_height = overlay.shape[0]

        y = 0


    # ---------------------------------------------
    # Crop right
    # ---------------------------------------------

    if x + overlay_width > frame_width:

        overlay = overlay[
            :,
            :frame_width - x
        ]

        overlay_width = overlay.shape[1]


    # ---------------------------------------------
    # Crop bottom
    # ---------------------------------------------

    if y + overlay_height > frame_height:

        overlay = overlay[
            :frame_height - y,
            :
        ]

        overlay_height = overlay.shape[0]


    if overlay_width <= 0 or overlay_height <= 0:

        return


    # ---------------------------------------------
    # Alpha channel
    # ---------------------------------------------

    if overlay.shape[2] == 4:

        alpha = (
            overlay[:, :, 3]
            .astype(float)
            / 255.0
        )

        alpha = alpha[:, :, None]

        foreground = (
            overlay[:, :, :3]
            .astype(float)
        )

        background = (
            frame[
                y:y + overlay_height,
                x:x + overlay_width
            ]
            .astype(float)
        )

        result = (
            alpha * foreground
            +
            (1 - alpha) * background
        )

        frame[
            y:y + overlay_height,
            x:x + overlay_width
        ] = result.astype("uint8")

    else:

        frame[
            y:y + overlay_height,
            x:x + overlay_width
        ] = overlay[:, :, :3]


# =========================================================
# VIDEO PROCESSOR
# =========================================================

class SpectaclesProcessor(VideoProcessorBase):

    def __init__(self):

        self.glasses = glasses_image.copy()

        self.size_factor = size_factor

        self.vertical_offset = vertical_offset

        self.face_mesh = (
            mp.solutions.face_mesh.FaceMesh(

                max_num_faces=1,

                refine_landmarks=True,

                min_detection_confidence=0.5,

                min_tracking_confidence=0.5
            )
        )


    # =====================================================
    # PROCESS CAMERA FRAME
    # =====================================================

    def recv(self, frame):

        image = frame.to_ndarray(
            format="bgr24"
        )


        # ---------------------------------------------
        # Mirror webcam
        # ---------------------------------------------

        image = cv2.flip(
            image,
            1
        )


        frame_height, frame_width = image.shape[:2]


        # ---------------------------------------------
        # Convert BGR → RGB
        # ---------------------------------------------

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )


        # ---------------------------------------------
        # Face landmarks
        # ---------------------------------------------

        results = self.face_mesh.process(rgb)


        # =================================================
        # FACE DETECTED
        # =================================================

        if results.multi_face_landmarks:

            landmarks = (
                results.multi_face_landmarks[0]
            )


            # ---------------------------------------------
            # Eye landmarks
            # ---------------------------------------------

            left_eye = landmarks.landmark[33]

            right_eye = landmarks.landmark[263]


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


            # ---------------------------------------------
            # Eye distance
            # ---------------------------------------------

            eye_distance = math.sqrt(

                (right_x - left_x) ** 2
                +
                (right_y - left_y) ** 2
            )


            # ---------------------------------------------
            # Glasses width
            # ---------------------------------------------

            glasses_width = int(
                eye_distance * self.size_factor
            )


            if glasses_width < 20:

                glasses_width = 20


            # ---------------------------------------------
            # Original dimensions
            # ---------------------------------------------

            original_height, original_width = (
                self.glasses.shape[:2]
            )


            # ---------------------------------------------
            # Maintain aspect ratio
            # ---------------------------------------------

            glasses_height = int(
                glasses_width
                * original_height
                / original_width
            )


            if glasses_height < 10:

                glasses_height = 10


            # ---------------------------------------------
            # Resize glasses
            # ---------------------------------------------

            resized = cv2.resize(

                self.glasses,

                (
                    glasses_width,
                    glasses_height
                ),

                interpolation=cv2.INTER_AREA
            )


            # ---------------------------------------------
            # Calculate head angle
            # ---------------------------------------------

            angle = math.degrees(

                math.atan2(

                    right_y - left_y,

                    right_x - left_x
                )
            )


            # ---------------------------------------------
            # Rotate glasses
            # ---------------------------------------------

            rotated = rotate_image(
                resized,
                -angle
            )


            rotated_height, rotated_width = (
                rotated.shape[:2]
            )


            # ---------------------------------------------
            # Center between eyes
            # ---------------------------------------------

            center_x = int(
                (left_x + right_x) / 2
            )

            center_y = int(
                (left_y + right_y) / 2
            )


            # ---------------------------------------------
            # Vertical adjustment
            # ---------------------------------------------

            center_y += int(
                glasses_height
                * self.vertical_offset
            )


            # ---------------------------------------------
            # Final position
            # ---------------------------------------------

            x = int(
                center_x
                - rotated_width / 2
            )

            y = int(
                center_y
                - rotated_height / 2
            )


            # ---------------------------------------------
            # Put glasses on face
            # ---------------------------------------------

            overlay_glasses(
                image,
                rotated,
                x,
                y
            )


            # ---------------------------------------------
            # Status
            # ---------------------------------------------

            cv2.rectangle(
                image,
                (15, 15),
                (190, 55),
                (0, 180, 0),
                -1
            )

            cv2.putText(

                image,

                "Face Detected",

                (25, 42),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.65,

                (255, 255, 255),

                2
            )


        # =================================================
        # NO FACE
        # =================================================

        else:

            cv2.rectangle(
                image,
                (15, 15),
                (210, 55),
                (0, 0, 180),
                -1
            )

            cv2.putText(

                image,

                "Face Not Detected",

                (25, 42),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.65,

                (255, 255, 255),

                2
            )


        # =================================================
        # RETURN VIDEO FRAME
        # =================================================

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )


# =========================================================
# MAIN LAYOUT
# =========================================================

left_column, right_column = st.columns(
    [2.5, 1]
)


# =========================================================
# CAMERA
# =========================================================

with left_column:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("📷 Live Virtual Try-On")

    st.write(
        "Press START and allow camera access."
    )

    webrtc_streamer(

        key="spectacles-try-on",

        video_processor_factory=SpectaclesProcessor,

        media_stream_constraints={
            "video": True,
            "audio": False
        }
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# =========================================================
# INFORMATION PANEL
# =========================================================

with right_column:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("👓 Selected Style")

    st.write(
        f"### {selected_glasses}"
    )

    st.markdown(
        """
        <div class="info-box">
        <b>How to use:</b><br><br>
        1. Press START<br>
        2. Allow camera permission<br>
        3. Look at the camera<br>
        4. Move your head naturally<br>
        5. Change glasses from the sidebar
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    st.success(
        "✨ AI face tracking is active."
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# =========================================================
# FEATURES
# =========================================================

st.markdown("---")

st.subheader("✨ Features")

feature1, feature2, feature3 = st.columns(3)

with feature1:

    st.markdown(
        """
        ### 👀 Face Tracking

        MediaPipe detects facial
        landmarks in real time.
        """
    )


with feature2:

    st.markdown(
        """
        ### 👓 Virtual Try-On

        Spectacles automatically
        follow your eyes.
        """
    )


with feature3:

    st.markdown(
        """
        ### 🔄 Head Tracking

        Glasses rotate naturally
        when you move your head.
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#777;">
        <b>Virtual Spectacles Try-On</b><br>
        Built with Python • OpenCV • MediaPipe • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)