// Auto-generated. Do not edit!

// (in-package unity_robotics_demo_msgs.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

let geometry_msgs = _finder('geometry_msgs');

//-----------------------------------------------------------

class ObjectPoseServiceRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.object_name = null;
    }
    else {
      if (initObj.hasOwnProperty('object_name')) {
        this.object_name = initObj.object_name
      }
      else {
        this.object_name = '';
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type ObjectPoseServiceRequest
    // Serialize message field [object_name]
    bufferOffset = _serializer.string(obj.object_name, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type ObjectPoseServiceRequest
    let len;
    let data = new ObjectPoseServiceRequest(null);
    // Deserialize message field [object_name]
    data.object_name = _deserializer.string(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += _getByteLength(object.object_name);
    return length + 4;
  }

  static datatype() {
    // Returns string type for a service object
    return 'unity_robotics_demo_msgs/ObjectPoseServiceRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '2f12226348d323c2e5b2031b3da53f1b';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    string object_name
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new ObjectPoseServiceRequest(null);
    if (msg.object_name !== undefined) {
      resolved.object_name = msg.object_name;
    }
    else {
      resolved.object_name = ''
    }

    return resolved;
    }
};

class ObjectPoseServiceResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.object_pose = null;
    }
    else {
      if (initObj.hasOwnProperty('object_pose')) {
        this.object_pose = initObj.object_pose
      }
      else {
        this.object_pose = new geometry_msgs.msg.Pose();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type ObjectPoseServiceResponse
    // Serialize message field [object_pose]
    bufferOffset = geometry_msgs.msg.Pose.serialize(obj.object_pose, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type ObjectPoseServiceResponse
    let len;
    let data = new ObjectPoseServiceResponse(null);
    // Deserialize message field [object_pose]
    data.object_pose = geometry_msgs.msg.Pose.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 56;
  }

  static datatype() {
    // Returns string type for a service object
    return 'unity_robotics_demo_msgs/ObjectPoseServiceResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '4f370b349a20e2e0029b6c926ba1b874';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
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
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new ObjectPoseServiceResponse(null);
    if (msg.object_pose !== undefined) {
      resolved.object_pose = geometry_msgs.msg.Pose.Resolve(msg.object_pose)
    }
    else {
      resolved.object_pose = new geometry_msgs.msg.Pose()
    }

    return resolved;
    }
};

module.exports = {
  Request: ObjectPoseServiceRequest,
  Response: ObjectPoseServiceResponse,
  md5sum() { return '31d8db4c34c65d79af3330e45c30d801'; },
  datatype() { return 'unity_robotics_demo_msgs/ObjectPoseService'; }
};
