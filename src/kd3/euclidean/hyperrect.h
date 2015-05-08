/**
 *  @file
 *
 *  @date   Mar 26, 2011
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief
 */

#ifndef MPBLOCKS_KD_TREE_EUCLIDEAN_HYPERRECT_H_
#define MPBLOCKS_KD_TREE_EUCLIDEAN_HYPERRECT_H_

namespace mpblocks {
namespace kd_tree {
namespace euclidean {

/// an NDim dimensional hyperrectangle, represented as a min and max extent
/**
 *  implemented by storing the minimum and maximum extent of the hyper-rectangle
 *  in all dimensions
 */
template <class Traits>
struct HyperRect {
  typedef typename Traits::Format_t Format_t;
  typedef Eigen::Matrix<Format_t, Traits::NDim, 1> Vector_t;
  typedef Vector_t Point_t;

  Point_t minExt;  ///< minimum extent of hyper-rectangle
  Point_t maxExt;  ///< maximum extent of hyper-rectangle

  /// initialize min and max ext to be 0,0,...
  HyperRect();

  /// find the nearest point in the hyper-rectangle to the query point and
  /// return it's distance (squared)
  Format_t dist2(const Point_t& point);

  /// return the measure of the hypercube
  Format_t measure();
};

}  // namespace euclidean
}  // namespace kd_tree
}  // namespace mpblocks

#endif
