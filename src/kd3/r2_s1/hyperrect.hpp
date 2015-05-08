/**
 *  @file
 *
 *  @date   Mar 26, 2011
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief
 */

#ifndef MPBLOCKS_KD_TREE_R2_S1_HYPERRECT_HPP_
#define MPBLOCKS_KD_TREE_R2_S1_HYPERRECT_HPP_

#include <limits>

namespace mpblocks {
namespace kd_tree {
namespace r2_s1 {

template <class Traits>
HyperRect<Traits>::HyperRect() {
  if (Traits::NDim != Eigen::Dynamic) {
    minExt.fill(0);
    maxExt.fill(0);
  }
}

template <class Traits>
typename Traits::Format_t HyperRect<Traits>::measure() {
  Format_t s = 1.0;
  for (unsigned int i = 0; i < minExt.rows(); i++) s *= maxExt[i] - minExt[i];

  return s;
}

}  // namespace r2_s1
}  // namespace kd_tree
}  // namespace mpblocks

#endif
