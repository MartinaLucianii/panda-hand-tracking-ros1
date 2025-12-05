# panda-hand-tracking-ros1
Codici ROS1 per il controllo del manipolatore Franka Emika Panda in Gazebo tramite hand tracking 3D con Intel RealSense e MediaPipe.

# Controllo Franka Panda con Hand Tracking 3D (ROS1)

Questa repository contiene i codici sviluppati per controllare il manipolatore Franka Emika Panda in ambiente Gazebo (ROS1) utilizzando il tracciamento 3D della mano ottenuto da una camera Intel RealSense e MediaPipe.

## Contenuto

- `hand_tracker_3d.py` — Nodo ROS che rileva la mano, ricostruisce la posizione 3D e pubblica comandi di velocità.
- `arm_cartesian_controller.py` — Nodo ROS che converte i comandi in pose cartesiane per il manipolatore.
- `panda_only.launch` — Launch file per avviare Gazebo con Franka Panda e il controller cartesiano.

## Requisiti

- ROS1 Noetic
- `franka_gazebo`, `franka_example_controllers`
- Intel RealSense + `realsense2_camera`
- Python 3, OpenCV, MediaPipe
