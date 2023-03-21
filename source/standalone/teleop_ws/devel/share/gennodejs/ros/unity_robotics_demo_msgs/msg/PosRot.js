// Auto-generated. Do not edit!

// (in-package unity_robotics_demo_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

class PosRot {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.pos_x = null;
      this.pos_y = null;
      this.pos_z = null;
      this.rot_x = null;
      this.rot_y = null;
      this.rot_z = null;
      this.rot_w = null;
    }
    else {
      if (initObj.hasOwnProperty('pos_x')) {
        this.pos_x = initObj.pos_x
      }
      else {
        this.pos_x = 0.0;
      }
      if (initObj.hasOwnProperty('pos_y')) {
        this.pos_y = initObj.pos_y
      }
      else {
        this.pos_y = 0.0;
      }
      if (initObj.hasOwnProperty('pos_z')) {
        this.pos_z = initObj.pos_z
      }
      else {
        this.pos_z = 0.0;
      }
      if (initObj.hasOwnProperty('rot_x')) {
        this.rot_x = initObj.rot_x
      }
      else {
        this.rot_x = 0.0;
      }
      if (initObj.hasOwnProperty('rot_y')) {
        this.rot_y = initObj.rot_y
      }
      else {
        this.rot_y = 0.0;
      }
      if (initObj.hasOwnProperty('rot_z')) {
        this.rot_z = initObj.rot_z
      }
      else {
        this.rot_z = 0.0;
      }
      if (initObj.hasOwnProperty('rot_w')) {
        this.rot_w = initObj.rot_w
      }
      else {
        this.rot_w = 0.0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type PosRot
    // Serialize message field [pos_x]
    bufferOffset = _serializer.float32(obj.pos_x, buffer, bufferOffset);
    // Serialize message field [pos_y]
    bufferOffset = _serializer.float32(obj.pos_y, buffer, bufferOffset);
    // Serialize message field [pos_z]
    bufferOffset = _serializer.float32(obj.pos_z, buffer, bufferOffset);
    // Serialize message field [rot_x]
    bufferOffset = _serializer.float32(obj.rot_x, buffer, bufferOffset);
    // Serialize message field [rot_y]
    bufferOffset = _serializer.float32(obj.rot_y, buffer, bufferOffset);
    // Serialize message field [rot_z]
    bufferOffset = _serializer.float32(obj.rot_z, buffer, bufferOffset);
    // Serialize message field [rot_w]
    bufferOffset = _serializer.float32(obj.rot_w, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type PosRot
    let len;
    let data = new PosRot(null);
    // Deserialize message field [pos_x]
    data.pos_x = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [pos_y]
    data.pos_y = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [pos_z]
    data.pos_z = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [rot_x]
    data.rot_x = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [rot_y]
    data.rot_y = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [rot_z]
    data.rot_z = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [rot_w]
    data.rot_w = _deserializer.float32(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 28;
  }

  static datatype() {
    // Returns string type for a message object
    return 'unity_robotics_demo_msgs/PosRot';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '9869d8cce41ebbf62de929a1dd9fa7ce';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    float32 pos_x
    float32 pos_y
    float32 pos_z
    float32 rot_x
    float32 rot_y
    float32 rot_z
    float32 rot_w
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new PosRot(null);
    if (msg.pos_x !== undefined) {
      resolved.pos_x = msg.pos_x;
    }
    else {
      resolved.pos_x = 0.0
    }

    if (msg.pos_y !== undefined) {
      resolved.pos_y = msg.pos_y;
    }
    else {
      resolved.pos_y = 0.0
    }

    if (msg.pos_z !== undefined) {
      resolved.pos_z = msg.pos_z;
    }
    else {
      resolved.pos_z = 0.0
    }

    if (msg.rot_x !== undefined) {
      resolved.rot_x = msg.rot_x;
    }
    else {
      resolved.rot_x = 0.0
    }

    if (msg.rot_y !== undefined) {
      resolved.rot_y = msg.rot_y;
    }
    else {
      resolved.rot_y = 0.0
    }

    if (msg.rot_z !== undefined) {
      resolved.rot_z = msg.rot_z;
    }
    else {
      resolved.rot_z = 0.0
    }

    if (msg.rot_w !== undefined) {
      resolved.rot_w = msg.rot_w;
    }
    else {
      resolved.rot_w = 0.0
    }

    return resolved;
    }
};

module.exports = PosRot;
