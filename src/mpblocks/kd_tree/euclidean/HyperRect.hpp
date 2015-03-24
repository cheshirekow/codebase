/**
 *  @file
 *
 *  @date   Mar 26, 2011
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief
 */

#ifndef MPBLOCKS_KD_TREE_EUCLIDEAN_HYPERRECT_HPP_
#define MPBLOCKS_KD_TREE_EUCLIDEAN_HYPERRECT_HPP_

#include <limits>

namespace  mpblocks {
namespace   kd_tree {
namespace euclidean {


template < class Traits >
HyperRect<Traits>::HyperRect()
{
    minExt.fill(0);
    maxExt.fill(0);
}





template < class Traits >
typename Traits::Format_t
    HyperRect<Traits>::dist2(const Point_t& point)
{
    Format_t dist2 = 0;
    Format_t dist2i = 0;

    for (unsigned int i=0; i < point.rows(); i++)
    {
        if (point[i] < minExt[i])
            dist2i = minExt[i] - point[i];
        else if(point[i] > maxExt[i])
            dist2i = maxExt[i] - point[i];
        else
            dist2i = 0;

        dist2i *= dist2i;
        dist2 += dist2i;
    }

    return dist2;
}



template < class Traits >
typename Traits::Format_t
    HyperRect<Traits>::measure()
{
    Format_t s = 1.0;
    for(unsigned int i=0; i < minExt.rows(); i++)
        s *= maxExt[i] - minExt[i];

    return s;
}



} // namespace euclidean
} // namespace kd_tree
} // namespace mpblocks

#endif
