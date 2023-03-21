;; Auto-generated. Do not edit!


(when (boundp 'unity_robotics_demo_msgs::UnityColor)
  (if (not (find-package "UNITY_ROBOTICS_DEMO_MSGS"))
    (make-package "UNITY_ROBOTICS_DEMO_MSGS"))
  (shadow 'UnityColor (find-package "UNITY_ROBOTICS_DEMO_MSGS")))
(unless (find-package "UNITY_ROBOTICS_DEMO_MSGS::UNITYCOLOR")
  (make-package "UNITY_ROBOTICS_DEMO_MSGS::UNITYCOLOR"))

(in-package "ROS")
;;//! \htmlinclude UnityColor.msg.html


(defclass unity_robotics_demo_msgs::UnityColor
  :super ros::object
  :slots (_r _g _b _a ))

(defmethod unity_robotics_demo_msgs::UnityColor
  (:init
   (&key
    ((:r __r) 0)
    ((:g __g) 0)
    ((:b __b) 0)
    ((:a __a) 0)
    )
   (send-super :init)
   (setq _r (round __r))
   (setq _g (round __g))
   (setq _b (round __b))
   (setq _a (round __a))
   self)
  (:r
   (&optional __r)
   (if __r (setq _r __r)) _r)
  (:g
   (&optional __g)
   (if __g (setq _g __g)) _g)
  (:b
   (&optional __b)
   (if __b (setq _b __b)) _b)
  (:a
   (&optional __a)
   (if __a (setq _a __a)) _a)
  (:serialization-length
   ()
   (+
    ;; int32 _r
    4
    ;; int32 _g
    4
    ;; int32 _b
    4
    ;; int32 _a
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; int32 _r
       (write-long _r s)
     ;; int32 _g
       (write-long _g s)
     ;; int32 _b
       (write-long _b s)
     ;; int32 _a
       (write-long _a s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; int32 _r
     (setq _r (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; int32 _g
     (setq _g (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; int32 _b
     (setq _b (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; int32 _a
     (setq _a (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get unity_robotics_demo_msgs::UnityColor :md5sum-) "deccfa94a2f280f4c4cea955620ed984")
(setf (get unity_robotics_demo_msgs::UnityColor :datatype-) "unity_robotics_demo_msgs/UnityColor")
(setf (get unity_robotics_demo_msgs::UnityColor :definition-)
      "int32 r
int32 g
int32 b
int32 a

")



(provide :unity_robotics_demo_msgs/UnityColor "deccfa94a2f280f4c4cea955620ed984")


