
(cl:in-package :asdf)

(defsystem "unity_robotics_demo_msgs-srv"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :geometry_msgs-msg
               :unity_robotics_demo_msgs-msg
)
  :components ((:file "_package")
    (:file "ObjectPoseService" :depends-on ("_package_ObjectPoseService"))
    (:file "_package_ObjectPoseService" :depends-on ("_package"))
    (:file "PositionService" :depends-on ("_package_PositionService"))
    (:file "_package_PositionService" :depends-on ("_package"))
  ))