#!/usr/bin/env python3
"""
Nodo ROS: arm_cartesian_controller

Scopo
-----
- Riceve comandi di velocità cartesiana dall'hand tracker sul topic:
    /arm_cmd   (geometry_msgs/Twist)
- Integra queste velocità in una posa cartesiana dell'end-effector.
- Pubblica la posa target sul controller Franka:
    /cartesian_impedance_example_controller/equilibrium_pose (PoseStamped)

Note
----
- L'orientazione dell'end-effector viene letta una volta da
  /franka_state_controller/franka_states e poi mantenuta costante.
- Vengono applicati limiti di workspace, raggio massimo e passo massimo
  per rendere il movimento più sicuro e stabile.
"""

import rospy
import numpy as np
import tf.transformations as tr

from geometry_msgs.msg import Twist, PoseStamped
from franka_msgs.msg import FrankaState


class ArmCartesianController:
    def __init__(self):
        rospy.init_node("arm_cartesian_controller")

        # ========================
        # POSIZIONE INIZIALE (m)
        # (sarà sovrascritta da FrankaState se disponibile)
        # ========================
        self.x = rospy.get_param("~x0", 0.4)
        self.y = rospy.get_param("~y0", 0.0)
        self.z = rospy.get_param("~z0", 0.4)

        # Orientazione iniziale (quaternion)
        self.qx = 0.0
        self.qy = 0.0
        self.qz = 0.0
        self.qw = 1.0

        # ========================
        # WORKSPACE CARTESIANO (m)
        # ========================
        self.min_x = rospy.get_param("~min_x", 0.25)
        self.max_x = rospy.get_param("~max_x", 0.55)
        self.min_y = rospy.get_param("~min_y", -0.25)
        self.max_y = rospy.get_param("~max_y", 0.25)
        self.min_z = rospy.get_param("~min_z", 0.2)
        self.max_z = rospy.get_param("~max_z", 0.55)

        # ========================
        # LIMITE RADIALE (m)
        # ========================
        # opzionale: raggio massimo orizzontale dal basamento
        self.max_radius = rospy.get_param("~max_radius", 0.60)

        # ========================
        # LIMITI DI VELOCITÀ (m/s)
        # ========================
        self.max_vx = rospy.get_param("~max_vx", 0.10)
        self.max_vy = rospy.get_param("~max_vy", 0.10)
        self.max_vz = rospy.get_param("~max_vz", 0.10)

        # ========================
        # PASSO MASSIMO PER CICLO (m)
        # ========================
        self.max_step = rospy.get_param("~max_step", 0.005)

        # Timeout: se non arrivano comandi da un po', fermo la velocità
        self.timeout = rospy.get_param("~timeout", 0.3)

        # Stato comando
        self.last_cmd = Twist()
        self.last_cmd_time = rospy.Time.now()

        # Subscriber da hand tracker (/arm_cmd)
        self.sub_cmd = rospy.Subscriber(
            "/arm_cmd", Twist, self.cmd_cb, queue_size=1
        )

        # Publisher verso controller Franka
        self.pub_pose = rospy.Publisher(
            "/cartesian_impedance_example_controller/equilibrium_pose",
            PoseStamped,
            queue_size=1,
        )

        # Inizializza x,y,z,quat dalla pose attuale del robot (se disponibile)
        self.init_from_franka_state()

        # Timer d’integrazione (50 Hz)
        self.last_update = rospy.Time.now()
        rospy.Timer(rospy.Duration(0.02), self.update_pose)

        rospy.loginfo("ArmCartesianController avviato.")

    # ============
    # INIT DA FRANKA
    # ============

    def init_from_franka_state(self):
        """Legge una volta la pose EE iniziale da franka_state_controller."""
        try:
            rospy.loginfo("Attendo FrankaState iniziale...")
            msg = rospy.wait_for_message(
                "/franka_state_controller/franka_states",
                FrankaState,
                timeout=5.0,
            )

            # O_T_EE è una matrice 4x4 flatten row-major
            T = np.reshape(msg.O_T_EE, (4, 4)).T
            pos = T[:3, 3]
            q = tr.quaternion_from_matrix(T)
            q = q / np.linalg.norm(q)

            self.x, self.y, self.z = pos[0], pos[1], pos[2]
            self.qx, self.qy, self.qz, self.qw = q[0], q[1], q[2], q[3]

            rospy.loginfo(
                "Pose iniziale EE: x=%.3f y=%.3f z=%.3f",
                self.x, self.y, self.z
            )
        except Exception as e:
            rospy.logwarn(
                "Non sono riuscito a leggere FrankaState iniziale (%s). "
                "Uso i parametri x0,y0,z0 di default.", e
            )

    # ============
    # UTILITIES
    # ============

    def clamp(self, x, xmin, xmax):
        """Satura x nell’intervallo [xmin, xmax]."""
        return max(min(x, xmax), xmin)

    def saturate_vel(self, v, vmax):
        """Limita la velocità nell’intervallo [-vmax, vmax]."""
        return self.clamp(v, -vmax, vmax)

    # ============
    # CALLBACKS
    # ============

    def cmd_cb(self, msg: Twist):
        """Salva l'ultimo comando di velocità ricevuto da /arm_cmd."""
        self.last_cmd = msg
        self.last_cmd_time = rospy.Time.now()

    def update_pose(self, event):
        """Integra le velocità e pubblica la pose target per Franka."""
        now = rospy.Time.now()
        dt = (now - self.last_update).to_sec()
        if dt <= 0.0:
            return
        self.last_update = now

        # Se da troppo tempo non arrivano comandi -> fermo
        if (now - self.last_cmd_time).to_sec() > self.timeout:
            vx = vy = vz = 0.0
        else:
            # prendo le velocità dal comando e le saturo
            vx = self.saturate_vel(self.last_cmd.linear.x, self.max_vx)
            vy = self.saturate_vel(self.last_cmd.linear.y, self.max_vy)
            vz = self.saturate_vel(self.last_cmd.linear.z, self.max_vz)

        # Integrazione con limite sul passo massimo
        dx = vx * dt
        dy = vy * dt
        dz = vz * dt

        dx = self.clamp(dx, -self.max_step, self.max_step)
        dy = self.clamp(dy, -self.max_step, self.max_step)
        dz = self.clamp(dz, -self.max_step, self.max_step)

        self.x += dx
        self.y += dy
        self.z += dz

        # Limiti workspace (box cartesiano)
        self.x = self.clamp(self.x, self.min_x, self.max_x)
        self.y = self.clamp(self.y, self.min_y, self.max_y)
        self.z = self.clamp(self.z, self.min_z, self.max_z)

        # Limite radiale rispetto a panda_link0
        r = np.sqrt(self.x**2 + self.y**2)
        if r > self.max_radius:
            scale = self.max_radius / r
            self.x *= scale
            self.y *= scale

        # Costruisco il messaggio di PoseStamped
        pose_msg = PoseStamped()
        pose_msg.header.stamp = now
        pose_msg.header.frame_id = "panda_link0"

        pose_msg.pose.position.x = self.x
        pose_msg.pose.position.y = self.y
        pose_msg.pose.position.z = self.z

        # Uso SEMPRE l'orientazione iniziale letta dal robot
        pose_msg.pose.orientation.x = self.qx
        pose_msg.pose.orientation.y = self.qy
        pose_msg.pose.orientation.z = self.qz
        pose_msg.pose.orientation.w = self.qw

        self.pub_pose.publish(pose_msg)


if __name__ == "__main__":
    try:
        ArmCartesianController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
