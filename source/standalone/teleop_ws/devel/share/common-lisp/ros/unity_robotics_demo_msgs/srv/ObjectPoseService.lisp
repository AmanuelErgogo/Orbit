; Auto-generated. Do not edit!


(cl:in-package unity_robotics_demo_msgs-srv)


;//! \htmlinclude ObjectPoseService-request.msg.html

(cl:defclass <ObjectPoseService-request> (roslisp-msg-protocol:ros-message)
  ((object_name
    :reader object_name
    :initarg :object_name
    :type cl:string
    :initform ""))
)

(cl:defclass ObjectPoseService-request (<ObjectPoseService-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <ObjectPoseService-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'ObjectPoseService-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name unity_robotics_demo_msgs-srv:<ObjectPoseService-request> is deprecated: use unity_robotics_demo_msgs-srv:ObjectPoseService-request instead.")))

(cl:ensure-generic-function 'object_name-val :lambda-list '(m))
(cl:defmethod object_name-val ((m <ObjectPoseService-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader unity_robotics_demo_msgs-srv:object_name-val is deprecated.  Use unity_robotics_demo_msgs-srv:object_name instead.")
  (object_name m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <ObjectPoseService-request>) ostream)
  "Serializes a message object of type '<ObjectPoseService-request>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'object_name))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'object_name))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <ObjectPoseService-request>) istream)
  "Deserializes a message object of type '<ObjectPoseService-request>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'object_name) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'object_name) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<ObjectPoseService-request>)))
  "Returns string type for a service object of type '<ObjectPoseService-request>"
  "unity_robotics_demo_msgs/ObjectPoseServiceRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ObjectPoseService-request)))
  "Returns string type for a service object of type 'ObjectPoseService-request"
  "unity_robotics_demo_msgs/ObjectPoseServiceRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<ObjectPoseService-request>)))
  "Returns md5sum for a message object of type '<ObjectPoseService-request>"
  "31d8db4c34c65d79af3330e45c30d801")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'ObjectPoseService-request)))
  "Returns md5sum for a message object of type 'ObjectPoseService-request"
  "31d8db4c34c65d79af3330e45c30d801")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<ObjectPoseService-request>)))
  "Returns full string definition for message of type '<ObjectPoseService-request>"
  (cl:format cl:nil "string object_name~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'ObjectPoseService-request)))
  "Returns full string definition for message of type 'ObjectPoseService-request"
  (cl:format cl:nil "string object_name~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <ObjectPoseService-request>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'object_name))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <ObjectPoseService-request>))
  "Converts a ROS message object to a list"
  (cl:list 'ObjectPoseService-request
    (cl:cons ':object_name (object_name msg))
))
;//! \htmlinclude ObjectPoseService-response.msg.html

(cl:defclass <ObjectPoseService-response> (roslisp-msg-protocol:ros-message)
  ((object_pose
    :reader object_pose
    :initarg :object_pose
    :type geometry_msgs-msg:Pose
    :initform (cl:make-instance 'geometry_msgs-msg:Pose)))
)

(cl:defclass ObjectPoseService-response (<ObjectPoseService-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <ObjectPoseService-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'ObjectPoseService-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name unity_robotics_demo_msgs-srv:<ObjectPoseService-response> is deprecated: use unity_robotics_demo_msgs-srv:ObjectPoseService-response instead.")))

(cl:ensure-generic-function 'object_pose-val :lambda-list '(m))
(cl:defmethod object_pose-val ((m <ObjectPoseService-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader unity_robotics_demo_msgs-srv:object_pose-val is deprecated.  Use unity_robotics_demo_msgs-srv:object_pose instead.")
  (object_pose m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <ObjectPoseService-response>) ostream)
  "Serializes a message object of type '<ObjectPoseService-response>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'object_pose) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <ObjectPoseService-response>) istream)
  "Deserializes a message object of type '<ObjectPoseService-response>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'object_pose) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<ObjectPoseService-response>)))
  "Returns string type for a service object of type '<ObjectPoseService-response>"
  "unity_robotics_demo_msgs/ObjectPoseServiceResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ObjectPoseService-response)))
  "Returns string type for a service object of type 'ObjectPoseService-response"
  "unity_robotics_demo_msgs/ObjectPoseServiceResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<ObjectPoseService-response>)))
  "Returns md5sum for a message object of type '<ObjectPoseService-response>"
  "31d8db4c34c65d79af3330e45c30d801")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'ObjectPoseService-response)))
  "Returns md5sum for a message object of type 'ObjectPoseService-response"
  "31d8db4c34c65d79af3330e45c30d801")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<ObjectPoseService-response>)))
  "Returns full string definition for message of type '<ObjectPoseService-response>"
  (cl:format cl:nil "geometry_msgs/Pose object_pose~%~%~%================================================================================~%MSG: geometry_msgs/Pose~%# A representation of pose in free space, composed of position and orientation. ~%Point position~%Quaternion orientation~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'ObjectPoseService-response)))
  "Returns full string definition for message of type 'ObjectPoseService-response"
  (cl:format cl:nil "geometry_msgs/Pose object_pose~%~%~%================================================================================~%MSG: geometry_msgs/Pose~%# A representation of pose in free space, composed of position and orientation. ~%Point position~%Quaternion orientation~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <ObjectPoseService-response>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'object_pose))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <ObjectPoseService-response>))
  "Converts a ROS message object to a list"
  (cl:list 'ObjectPoseService-response
    (cl:cons ':object_pose (object_pose msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'ObjectPoseService)))
  'ObjectPoseService-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'ObjectPoseService)))
  'ObjectPoseService-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ObjectPoseService)))
  "Returns string type for a service object of type '<ObjectPoseService>"
  "unity_robotics_demo_msgs/ObjectPoseService")