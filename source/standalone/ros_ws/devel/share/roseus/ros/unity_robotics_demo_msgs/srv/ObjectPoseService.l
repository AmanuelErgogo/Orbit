;; Auto-generated. Do not edit!


(when (boundp 'unity_robotics_demo_msgs::ObjectPoseService)
  (if (not (find-package "UNITY_ROBOTICS_DEMO_MSGS"))
    (make-package "UNITY_ROBOTICS_DEMO_MSGS"))
  (shadow 'ObjectPoseService (find-package "UNITY_ROBOTICS_DEMO_MSGS")))
(unless (find-package "UNITY_ROBOTICS_DEMO_MSGS::OBJECTPOSESERVICE")
  (make-package "UNITY_ROBOTICS_DEMO_MSGS::OBJECTPOSESERVICE"))
(unless (find-package "UNITY_ROBOTICS_DEMO_MSGS::OBJECTPOSESERVICEREQUEST")
  (make-package "UNITY_ROBOTICS_DEMO_MSGS::OBJECTPOSESERVICEREQUEST"))
(unless (find-package "UNITY_ROBOTICS_DEMO_MSGS::OBJECTPOSESERVICERESPONSE")
  (make-package "UNITY_ROBOTICS_DEMO_MSGS::OBJECTPOSESERVICERESPONSE"))

(in-package "ROS")



(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))


(defclass unity_robotics_demo_msgs::ObjectPoseServiceRequest
  :super ros::object
  :slots (_object_name ))

(defmethod unity_robotics_demo_msgs::ObjectPoseServiceRequest
  (:init
   (&key
    ((:object_name __object_name) "")
    )
   (send-super :init)
   (setq _object_name (string __object_name))
   self)
  (:object_name
   (&optional __object_name)
   (if __object_name (setq _object_name __object_name)) _object_name)
  (:serialization-length
   ()
   (+
    ;; string _object_name
    4 (length _object_name)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _object_name
       (write-long (length _object_name) s) (princ _object_name s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _object_name
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _object_name (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass unity_robotics_demo_msgs::ObjectPoseServiceResponse
  :super ros::object
  :slots (_object_pose ))

(defmethod unity_robotics_demo_msgs::ObjectPoseServiceResponse
  (:init
   (&key
    ((:object_pose __object_pose) (instance geometry_msgs::Pose :init))
    )
   (send-super :init)
   (setq _object_pose __object_pose)
   self)
  (:object_pose
   (&rest __object_pose)
   (if (keywordp (car __object_pose))
       (send* _object_pose __object_pose)
     (progn
       (if __object_pose (setq _object_pose (car __object_pose)))
       _object_pose)))
  (:serialization-length
   ()
   (+
    ;; geometry_msgs/Pose _object_pose
    (send _object_pose :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; geometry_msgs/Pose _object_pose
       (send _object_pose :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; geometry_msgs/Pose _object_pose
     (send _object_pose :deserialize buf ptr-) (incf ptr- (send _object_pose :serialization-length))
   ;;
   self)
  )

(defclass unity_robotics_demo_msgs::ObjectPoseService
  :super ros::object
  :slots ())

(setf (get unity_robotics_demo_msgs::ObjectPoseService :md5sum-) "31d8db4c34c65d79af3330e45c30d801")
(setf (get unity_robotics_demo_msgs::ObjectPoseService :datatype-) "unity_robotics_demo_msgs/ObjectPoseService")
(setf (get unity_robotics_demo_msgs::ObjectPoseService :request) unity_robotics_demo_msgs::ObjectPoseServiceRequest)
(setf (get unity_robotics_demo_msgs::ObjectPoseService :response) unity_robotics_demo_msgs::ObjectPoseServiceResponse)

(defmethod unity_robotics_demo_msgs::ObjectPoseServiceRequest
  (:response () (instance unity_robotics_demo_msgs::ObjectPoseServiceResponse :init)))

(setf (get unity_robotics_demo_msgs::ObjectPoseServiceRequest :md5sum-) "31d8db4c34c65d79af3330e45c30d801")
(setf (get unity_robotics_demo_msgs::ObjectPoseServiceRequest :datatype-) "unity_robotics_demo_msgs/ObjectPoseServiceRequest")
(setf (get unity_robotics_demo_msgs::ObjectPoseServiceRequest :definition-)
      "string object_name
---
geometry_msgs/Pose object_pose


================================================================================
MSG: geometry_msgs/Pose
# A representation of pose in free space, composed of position and orientation. 
Point position
Quaternion orientation

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z

================================================================================
MSG: geometry_msgs/Quaternion
# This represents an orientation in free space in quaternion form.

float64 x
float64 y
float64 z
float64 w
")

(setf (get unity_robotics_demo_msgs::ObjectPoseServiceResponse :md5sum-) "31d8db4c34c65d79af3330e45c30d801")
(setf (get unity_robotics_demo_msgs::ObjectPoseServiceResponse :datatype-) "unity_robotics_demo_msgs/ObjectPoseServiceResponse")
(setf (get unity_robotics_demo_msgs::ObjectPoseServiceResponse :definition-)
      "string object_name
---
geometry_msgs/Pose object_pose


================================================================================
MSG: geometry_msgs/Pose
# A representation of pose in free space, composed of position and orientation. 
Point position
Quaternion orientation

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z

================================================================================
MSG: geometry_msgs/Quaternion
# This represents an orientation in free space in quaternion form.

float64 x
float64 y
float64 z
float64 w
")



(provide :unity_robotics_demo_msgs/ObjectPoseService "31d8db4c34c65d79af3330e45c30d801")


