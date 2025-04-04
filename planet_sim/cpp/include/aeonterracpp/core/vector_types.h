#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>
#include <vector>

namespace aeonterracpp {

/**
 * @brief 3D vector with float precision
 * 
 * Used for most geometry operations where double precision is not required.
 */
using Vec3f = Eigen::Vector3f;

/**
 * @brief 3D vector with double precision
 * 
 * Used for operations requiring higher precision, like coordinate transformations.
 */
using Vec3d = Eigen::Vector3d;

/**
 * @brief 3D vector with integer components
 * 
 * Used for face indices and other integer triples.
 */
using Vec3i = Eigen::Vector3i;

/**
 * @brief 4D vector with float precision
 * 
 * Used for homogeneous coordinates and RGBA colors.
 */
using Vec4f = Eigen::Vector4f;

/**
 * @brief Quaternion for rotations
 * 
 * Used for representing rotations in 3D space.
 */
using Quaternion = Eigen::Quaternionf;

/**
 * @brief 3x3 matrix with float precision
 * 
 * Used for transformations and rotations.
 */
using Matrix3f = Eigen::Matrix3f;

/**
 * @brief 4x4 matrix with float precision
 * 
 * Used for homogeneous transformations.
 */
using Matrix4f = Eigen::Matrix4f;

/**
 * @brief Convenience function to convert degrees to radians
 * 
 * @param degrees Angle in degrees
 * @return Angle in radians
 */
inline float toRadians(float degrees) {
    return degrees * 0.01745329251994329576923690768489f; // PI/180
}

/**
 * @brief Convenience function to convert radians to degrees
 * 
 * @param radians Angle in radians
 * @return Angle in degrees
 */
inline float toDegrees(float radians) {
    return radians * 57.295779513082320876798154814105f; // 180/PI
}

/**
 * @brief Bounding box in lat/lon coordinates
 */
struct BoundingBox {
    float latMin;
    float lonMin;
    float latMax;
    float lonMax;
    
    BoundingBox() : latMin(-90.0f), lonMin(-180.0f), latMax(90.0f), lonMax(180.0f) {}
    
    BoundingBox(float _latMin, float _lonMin, float _latMax, float _lonMax)
        : latMin(_latMin), lonMin(_lonMin), latMax(_latMax), lonMax(_lonMax) {}
        
    bool contains(float lat, float lon) const {
        return lat >= latMin && lat <= latMax && lon >= lonMin && lon <= lonMax;
    }
    
    float width() const { return lonMax - lonMin; }
    float height() const { return latMax - latMin; }
};

/**
 * @brief Axis-aligned bounding box in 3D space
 */
struct AABB {
    Vec3f min;
    Vec3f max;
    
    AABB() : min(Vec3f::Zero()), max(Vec3f::Zero()) {}
    
    AABB(const Vec3f& _min, const Vec3f& _max) : min(_min), max(_max) {}
    
    bool contains(const Vec3f& point) const {
        return (point.x() >= min.x() && point.x() <= max.x() &&
                point.y() >= min.y() && point.y() <= max.y() &&
                point.z() >= min.z() && point.z() <= max.z());
    }
    
    bool intersects(const AABB& other) const {
        return (min.x() <= other.max.x() && max.x() >= other.min.x() &&
                min.y() <= other.max.y() && max.y() >= other.min.y() &&
                min.z() <= other.max.z() && max.z() >= other.min.z());
    }
    
    Vec3f center() const {
        return (min + max) * 0.5f;
    }
    
    Vec3f size() const {
        return max - min;
    }
};

} // namespace aeonterracpp