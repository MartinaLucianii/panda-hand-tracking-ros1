#!/usr/bin/env python3
"""
Nodo ROS: hand_tracker_3d

Scopo
-----
- Acquisisce immagini RGB + depth da una Intel RealSense.
- Usa MediaPipe Hands per stimare i landmark della mano.
- Ricostruisce la posizione 3D del centro della mano.
- Riconosce gesti semplici:
    - "OPEN": mano aperta → attiva la modalità FOLLOW
    - "FIST": pugno chiuso → ferma il robot (STOP)
- Converte il movimento della mano in un comando di velocità (Twist)
  e lo pubblica su /arm_cmd per il controller cartesiano.

Topic I/O principali
--------------------
Sub:
  - /camera/color/image_raw
  - /camera/aligned_depth_to_color/image_raw
  - /camera/color/camera_info

Pub:
  - /hand_tracker/image_annotated (Image)
  - /hand_tracker/hand_center    (PointStamped)
  - /hand_tracker/landmarks      (PoseArray)
  - /arm_cmd                     (Twist)
"""

import rospy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped, Pose, PoseArray, Twist
from cv_bridge import CvBridge
import numpy as np
import cv2
import mediapipe as mp


class HandTracker3D:
    def __init__(self):
        rospy.init_node("hand_tracker_3d", anonymous=True)
        self.bridge = CvBridge()

        # ===================== PARAMETRI / TOPIC =====================

        # Topics camera
        self.rgb_topic = rospy.get_param(
            "~rgb_topic", "/camera/color/image_raw"
        )
        self.depth_topic = rospy.get_param(
            "~depth_topic", "/camera/aligned_depth_to_color/image_raw"
        )
        self.info_topic = rospy.get_param(
            "~info_topic", "/camera/color/camera_info"
        )
        # Topic di uscita comandi verso il braccio
        self.cmd_topic = rospy.get_param("~cmd_topic", "/arm_cmd")

        # Publisher
        self.pub_annot = rospy.Publisher(
            "/hand_tracker/image_annotated", Image, queue_size=1
        )
        self.pub_center = rospy.Publisher(
            "/hand_tracker/hand_center", PointStamped, queue_size=1
        )
        self.pub_lm = rospy.Publisher(
            "/hand_tracker/landmarks", PoseArray, queue_size=1
        )
        self.pub_cmd = rospy.Publisher(
            self.cmd_topic, Twist, queue_size=1
        )

        # Intrinseci camera
        self.fx = self.fy = self.cx = self.cy = None

        # Buffer ultima immagine di profondità
        self.depth_img = None

        # ==== stato per il controllo del robot ====
        self.prev_center = None        # (X, Y, Z) della mano al frame precedente
        self.follow_mode = False       # False = STOP, True = FOLLOW

        # Gain per convertire spostamento mano -> velocità robot
        self.GAIN_X = rospy.get_param("~gain_x", 12.0)
        self.GAIN_Y = rospy.get_param("~gain_y", 12.0)
        self.GAIN_Z = rospy.get_param("~gain_z", 12.0)
        # dead-zone molto piccola (1 mm)
        self.DEAD_ZONE = rospy.get_param("~dead_zone", 0.001)

        # Filtro 3D su posizione e "velocità" mano
        self.center_filt_3d = None                    # [X,Y,Z] filtrati
        self.vel_filt_3d = np.zeros(3, dtype=float)   # "velocità" filtrata
        self.ALPHA_POS_3D = rospy.get_param("~alpha_pos_3d", 0.6)
        self.ALPHA_VEL_3D = rospy.get_param("~alpha_vel_3d", 0.4)

        # ====== TRAIETTORIA (plot del movimento mano in pixel) ======
        self.TRAJ_SIZE = rospy.get_param("~traj_size", 400)   # dimensione canvas (pixel)
        self.traj_img = np.ones(
            (self.TRAJ_SIZE, self.TRAJ_SIZE, 3), dtype=np.uint8
        ) * 255
        self.prev_traj_pt = None         # punto precedente sul canvas
        self.ALPHA_POS_2D = rospy.get_param("~alpha_pos_2d", 0.4)  # filtro posizione 2D
        self.center2d_filtered = None    # [u, v] filtrati

        # MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_style = mp.solutions.drawing_styles

        # Subscriptions
        rospy.Subscriber(
            self.info_topic, CameraInfo, self.info_cb, queue_size=1
        )
        rospy.Subscriber(
            self.depth_topic,
            Image,
            self.depth_cb,
            queue_size=1,
            buff_size=2**24,
        )
        rospy.Subscriber(
            self.rgb_topic,
            Image,
            self.image_cb,
            queue_size=1,
            buff_size=2**24,
        )

        rospy.loginfo(
            "HandTracker3D pronto. Apri rqt_image_view su /hand_tracker/image_annotated"
        )
        rospy.spin()

    # ===================== CALLBACK CAMERA INFO =====================

    def info_cb(self, msg: CameraInfo):
        """
        Salva gli intrinseci della camera.

        Attenzione: l'immagine RGB viene flippata orizzontalmente,
        quindi aggiorniamo cx in modo coerente.
        """
        # Pinhole intrinsics originali
        self.fx = msg.K[0]
        self.fy = msg.K[4]
        cx_orig = msg.K[2]
        self.cy = msg.K[5]

        # Se stai flippando orizzontalmente, il nuovo centro cx' è:
        width = msg.width
        self.cx = (width - 1) - cx_orig

    # ========================= DEPTH CALLBACK ========================

    def depth_cb(self, msg: Image):
        """Salva l'ultima immagine di profondità, flippata come l'RGB."""
        try:
            d = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            d = cv2.flip(d, 1)
            self.depth_img = d
        except Exception as e:
            rospy.logwarn("Depth convert error: %s", e)

    # ==================== GESTIONE DITA / GESTI ======================

    def count_extended_fingers(self, hand_lms):
        """
        Conta quante dita sono "alzate".

        Usa solo la coordinata y normalizzata per capire se il tip è sopra
        l'articolazione PIP. Ignora il pollice per semplicità.
        """
        lms = hand_lms.landmark
        fingers = 0

        # Indice: tip 8, pip 6
        if lms[8].y < lms[6].y:
            fingers += 1
        # Medio: tip 12, pip 10
        if lms[12].y < lms[10].y:
            fingers += 1
        # Anulare: tip 16, pip 14
        if lms[16].y < lms[14].y:
            fingers += 1
        # Mignolo: tip 20, pip 18
        if lms[20].y < lms[18].y:
            fingers += 1

        return fingers

    def classify_gesture(self, hand_lms):
        """Ritorna una label semplice: OPEN / FIST / OTHER."""
        fingers = self.count_extended_fingers(hand_lms)
        if fingers >= 3:
            return "OPEN"
        elif fingers == 0:
            return "FIST"
        else:
            return "OTHER"

    # ==== comando per il manipolatore da movimento 3D ====

    def publish_cmd_from_motion(self, X, Y, Z):
        """
        Converte il movimento 3D della mano in un comando di velocità Twist.

        - Se follow_mode è True → usa lo spostamento filtrato della mano.
        - Se follow_mode è False → pubblica velocità nulla e azzera il filtro.
        """
        twist = Twist()

        if self.prev_center is not None and self.follow_mode:
            X_prev, Y_prev, Z_prev = self.prev_center

            # Spostamento "grezzo" da frame precedente
            dX = X - X_prev   # destra/sinistra in camera
            dY = Y - Y_prev   # su/giù in camera
            dZ = Z - Z_prev   # avanti/indietro in camera

            # Dead-zone
            if abs(dX) < self.DEAD_ZONE:
                dX = 0.0
            if abs(dY) < self.DEAD_ZONE:
                dY = 0.0
            if abs(dZ) < self.DEAD_ZONE:
                dZ = 0.0

            # =========================
            # FILTRO 3D SU POS E "VEL"
            # =========================
            # Aggiorno posizione filtrata
            if self.center_filt_3d is None:
                self.center_filt_3d = np.array([X, Y, Z], dtype=float)
            else:
                curr = np.array([X, Y, Z], dtype=float)
                self.center_filt_3d = (
                    self.ALPHA_POS_3D * self.center_filt_3d
                    + (1.0 - self.ALPHA_POS_3D) * curr
                )

            # "Velocità" stimata (differenza tra posizioni filtrate)
            dX_f = self.center_filt_3d[0] - X_prev
            dY_f = self.center_filt_3d[1] - Y_prev
            dZ_f = self.center_filt_3d[2] - Z_prev

            vel_raw = np.array([dX_f, dY_f, dZ_f], dtype=float)
            self.vel_filt_3d = (
                self.ALPHA_VEL_3D * self.vel_filt_3d
                + (1.0 - self.ALPHA_VEL_3D) * vel_raw
            )

            dX_f, dY_f, dZ_f = self.vel_filt_3d

            # === MAPPING CAMERA -> ROBOT (aggiustabile) ===
            # Mano avanti (aumenta Z_cam) -> robot avanti (+X_robot)
            vx = self.GAIN_X * dZ_f

            # Mano a destra (aumenta X_cam) -> robot a destra (-Y_robot)
            vy = -self.GAIN_Y * dX_f

            # Mano in alto (Y_cam diminuisce) -> robot in alto (+Z_robot)
            vz = -self.GAIN_Z * dY_f

            twist.linear.x = vx
            twist.linear.y = vy
            twist.linear.z = vz
        else:
            # Modalità STOP
            twist.linear.x = 0.0
            twist.linear.y = 0.0
            twist.linear.z = 0.0
            self.vel_filt_3d[:] = 0.0

        self.pub_cmd.publish(twist)
        # aggiorno prev_center con la POSIZIONE FILTRATA (se presente)
        if self.center_filt_3d is not None:
            Xs, Ys, Zs = self.center_filt_3d
        else:
            Xs, Ys, Zs = X, Y, Z
        self.prev_center = (Xs, Ys, Zs)

    # ===================== TRAIETTORIA 2D (PLOT) =====================

    def update_trajectory_2d(self, u, v, img_w, img_h):
        """
        u, v: coordinate pixel del centro mano nell'immagine (dopo flip).
        img_w, img_h: dimensioni dell'immagine.

        Disegna la traiettoria su self.traj_img (normalizzata a TRAJ_SIZE).
        """
        # normalizza da [0..w]x[0..h] a [0..TRAJ_SIZE]
        px = int(np.clip(u / float(img_w), 0.0, 1.0) * (self.TRAJ_SIZE - 1))
        py = int(np.clip(v / float(img_h), 0.0, 1.0) * (self.TRAJ_SIZE - 1))

        # se ho un punto precedente, disegno una linea
        if self.prev_traj_pt is not None:
            cv2.line(self.traj_img, self.prev_traj_pt, (px, py), (0, 0, 255), 2)
        else:
            cv2.circle(self.traj_img, (px, py), 3, (0, 0, 255), -1)

        self.prev_traj_pt = (px, py)

    # ========================== IMAGE CALLBACK =======================

    def image_cb(self, msg: Image):
        """Callback principale: elabora l’immagine RGB."""
        if self.fx is None or self.depth_img is None:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            # flip come nel codice originale (vista "specchio")
            frame = cv2.flip(frame, 1)
        except Exception as e:
            rospy.logerr("RGB convert error: %s", e)
            return

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        res = self.hands.process(rgb)
        annotated = frame.copy()
        poses = PoseArray()
        poses.header = msg.header

        if res.multi_hand_landmarks:
            handed = (
                res.multi_handedness
                if hasattr(res, "multi_handedness")
                else [None] * len(res.multi_hand_landmarks)
            )

            for idx, (hand_lms, hinfo) in enumerate(
                zip(res.multi_hand_landmarks, handed)
            ):
                # --- landmarks e connessioni
                self.mp_draw.draw_landmarks(
                    annotated,
                    hand_lms,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_style.get_default_hand_landmarks_style(),
                    self.mp_style.get_default_hand_connections_style(),
                )

                # --- bbox in pixel
                xs = [int(np.clip(lm.x * w, 0, w - 1)) for lm in hand_lms.landmark]
                ys = [int(np.clip(lm.y * h, 0, h - 1)) for lm in hand_lms.landmark]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                pad = 10
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(w - 1, x2 + pad)
                y2 = min(h - 1, y2 + pad)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)

                # --- centro bbox in pixel (per testo/overlay e TRAIETTORIA 2D)
                uc = int((x1 + x2) / 2)
                vc = int((y1 + y2) / 2)

                # filtro 2D per rendere la traiettoria più liscia
                center2d = np.array([uc, vc], dtype=float)
                if self.center2d_filtered is None:
                    self.center2d_filtered = center2d.copy()
                else:
                    self.center2d_filtered = (
                        self.ALPHA_POS_2D * self.center2d_filtered
                        + (1.0 - self.ALPHA_POS_2D) * center2d
                    )
                u_f, v_f = self.center2d_filtered.astype(int)

                # puntino sul centro filtrato
                cv2.circle(annotated, (u_f, v_f), 5, (0, 0, 255), -1)

                # profondità e deprojection per il controllo 3D
                Zm = self.depth_at(v_f, u_f)
                X, Y = self.deproject(u_f, v_f, Zm)

                # gesto mano (OPEN / FIST / OTHER)
                gesture = self.classify_gesture(hand_lms)

                # testo etichetta
                label = "Hand"
                score = None
                if hinfo and len(hinfo.classification) > 0:
                    label = hinfo.classification[0].label  # "Left" / "Right"
                    score = hinfo.classification[0].score

                text = f"{label}"
                if score is not None:
                    text += f" {score:.2f}"
                if not np.isnan(Zm):
                    text += f" Z={Zm:.2f}m"
                text += f" G={gesture}"

                cv2.putText(
                    annotated,
                    text,
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                # logica start/stop in base al gesto
                if gesture == "OPEN":
                    self.follow_mode = True
                elif gesture == "FIST":
                    self.follow_mode = False

                # pubblicazione di tutti i 21 landmark in PoseArray
                for lm in hand_lms.landmark:
                    uu = int(np.clip(lm.x * w, 0, w - 1))
                    vv = int(np.clip(lm.y * h, 0, h - 1))
                    Zm_i = self.depth_at(vv, uu)
                    Xi, Yi = self.deproject(uu, vv, Zm_i)
                    p = Pose()
                    p.position.x = Xi
                    p.position.y = Yi
                    p.position.z = Zm_i
                    p.orientation.w = 1.0
                    poses.poses.append(p)

                # pub. centro + comandi solo dalla prima mano
                if idx == 0:
                    # pubblica centro 3D per debug
                    ps = PointStamped()
                    ps.header = msg.header
                    ps.point.x = X
                    ps.point.y = Y
                    ps.point.z = Zm
                    self.pub_center.publish(ps)

                    # comando manipolatore (se c'è distanza valida)
                    if not np.isnan(Zm):
                        self.publish_cmd_from_motion(X, Y, Zm)

                    # aggiornamento TRAIETTORIA 2D usando il centro filtrato
                    self.update_trajectory_2d(u_f, v_f, w, h)

        # Pub landmarks e immagine annotata
        self.pub_lm.publish(poses)
        try:
            self.pub_annot.publish(self.bridge.cv2_to_imgmsg(annotated, "bgr8"))
        except Exception as e:
            rospy.logwarn("Annot publish error: %s", e)

        # Finestre di debug locali (se hai un display)
        try:
            mode_text = "FOLLOW" if self.follow_mode else "STOP"
            cv2.putText(
                annotated,
                f"MODE: {mode_text}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Hand + Depth (CPU)", annotated)
            cv2.imshow("Hand Trajectory", self.traj_img)
            cv2.waitKey(1)
        except Exception:
            # Se non c'è display (es. su server remoto) ignora l'errore
            pass

    # ======================= UTILS DEPTH / 3D =======================

    def depth_at(self, v, u):
        """Ritorna la profondità in metri leggendo la depth 16UC1 (mm)."""
        if self.depth_img is None:
            return float("nan")
        if (0 <= v < self.depth_img.shape[0]) and (0 <= u < self.depth_img.shape[1]):
            z_mm = int(self.depth_img[v, u])
            if z_mm > 0:
                return z_mm / 1000.0  # mm -> m
        return float("nan")

    def deproject(self, u, v, Z):
        """Pinhole back-projection: pixel(u,v), profondità Z (m) -> X,Y (m)."""
        if np.isnan(Z):
            return float("nan"), float("nan")
        X = (u - self.cx) * Z / self.fx
        Y = (v - self.cy) * Z / self.fy
        return X, Y


if __name__ == "__main__":
    try:
        HandTracker3D()
    except rospy.ROSInterruptException:
        pass
