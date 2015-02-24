
#ifndef MPBLOCKS_LINALG_EULER_ANGLES_H_
#define MPBLOCKS_LINALG_EULER_ANGLES_H_

namespace mpblocks {
namespace linalg {

namespace euler {

/// A vector of euler angles.
/**
 *  The convention is that rotations are applied in the reverse order that they
 *  are specified in the vector. For instance the orientation described by an
 *  euler::Angles<Scalar,0,1,2>(phi, theta, psi) vector corresponds to an
 *  orientation described by a rotation of psi about the z-axis (2-axis),
 *  followed by a rotation of theta about the new y-axis (1-axis), followed by
 *  a rotation of phi about the new x-axis (0-axis).
 */
template <typename Scalar, int Axis1, int Axis2, int Axis3>
class Angles : public Matrix<Scalar, 3, 1> {
 public:
  typedef Matrix<Scalar, 3, 1> MatrixBase;

  Angles() {}

  /// We require at least two values so as to disambiguate the constructor
  /// from the construct-by-rvalue constructor
  Angles(Scalar a, Scalar b, Scalar c)
      : MatrixBase(a, b, c) {
  }

  Angles(const Quaternion<Scalar>& q) {
#if MPBLOCKS_USE_STATIC_ASSERT
    static_assert(
        _AnglesFromQuaternion<Scalar, Axis1, Axis2, Axis3>::HAS_SPECIALIZATION,
        "There is no conversion from quaternions to this euler angle set");
#endif
    _AnglesFromQuaternion<Scalar, Axis1, Axis2, Axis3>::Build(q, this);
  }

  friend void EulerAnglesToQuaternion<Scalar, Axis1, Axis2, Axis3>(
      const euler::Angles<Scalar, Axis1, Axis2, Axis3>& euler,
      Quaternion<Scalar>* q);
};

typedef Angles<double,_1,_2,_1> Angles121d;
typedef Angles<double,_1,_2,_3> Angles123d;
typedef Angles<double,_3,_1,_3> Angles313d;

typedef Angles<float,_1,_2,_1> Angles121f;
typedef Angles<float,_1,_2,_3> Angles123f;
typedef Angles<float,_3,_1,_3> Angles313f;

}  // namespace euler
}  // namespace linalg
}  // namespace mpblocks

#endif  // MPBLOCKS_LINALG_EULER_ANGLES_H_
