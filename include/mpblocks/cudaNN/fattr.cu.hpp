/*
 *  Copyright (C) 2012 Josh Bialkowski (jbialk@mit.edu)
 *
 *  This file is part of mpblocks.
 *
 *  mpblocks is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  mpblocks is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with mpblocks.  If not, see <http://www.gnu.org/licenses/>.
 */
/**
 *  @file   /home/josh/Codes/cpp/mpblocks2/cuda_nn/include/mpblocks/cudaNN/fattr.cu.hpp
 *
 *  @date   Oct 7, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDANN_FATTR_CU_HPP_
#define MPBLOCKS_CUDANN_FATTR_CU_HPP_

#include <mpblocks/cuda/bitonic.cu.hpp>
#include <mpblocks/cuda/bitonic/fattr.cu.hpp>
#include <mpblocks/cudaNN/fattr.h>
#include <mpblocks/cudaNN/kernels.cu.hpp>

namespace mpblocks {
namespace   cudaNN {

template <typename Format_t, unsigned int NDim,bool Enabled>
struct SE3FAttr
{
    static void append(fattrMap_t& map){}
};

template <typename Format_t, unsigned int NDim>
struct SE3FAttr<Format_t,NDim,true>
{
    static void append(fattrMap_t& map)
    {
        map["se3_distance"]
           .getFrom( &kernels::se3_distance<Format_t,NDim> );
    }
};

template <typename Format_t, unsigned int NDim,bool Enabled>
struct SO3FAttr
{
    static void append(fattrMap_t& map){}
};

template <typename Format_t, unsigned int NDim>
struct SO3FAttr<Format_t,NDim,true>
{
    static void append(fattrMap_t& map)
    {
        typedef QueryPoint<Format_t,NDim>     QP;
        typedef RectangleQuery<Format_t,NDim> QR;
        typedef unsigned int                  uint;
        typedef void (*point_dist_fn)(QP,Format_t*,uint,Format_t*,uint,uint);
        typedef void (*rect_dist_fn) (QR,Format_t*);

        point_dist_fn so3_point_dist =
                &kernels::so3_distance<false,Format_t,NDim>;
        point_dist_fn so3_point_pseudo_dist =
                &kernels::so3_distance<true,Format_t,NDim>;
        rect_dist_fn  so3_rect_dist =
                &kernels::so3_distance<false,Format_t,NDim>;
        rect_dist_fn  so3_rect_pseudo_dist =
                &kernels::so3_distance<true,Format_t,NDim>;

        map["se3_point_distance"]       .getFrom(so3_point_dist);
        map["se3_point_pseudo_distance"].getFrom(so3_point_pseudo_dist);
        map["se3_rect_distance"]        .getFrom(so3_rect_dist);
        map["se3_rect_pseudo_distance"] .getFrom(so3_rect_pseudo_dist);
    }
};



template <typename Format_t, unsigned int NDim>
void get_fattr( fattrMap_t& map )
{
    typedef QueryPoint<Format_t,NDim>     QP;
    typedef RectangleQuery<Format_t,NDim> QR;
    typedef unsigned int                  uint;
    typedef void (*point_dist_fn)(QP,Format_t*,uint,Format_t*,uint,uint);
    typedef void (*rect_dist_fn) (QR,Format_t*);


    point_dist_fn euclidean_point_dist =
            &kernels::euclidean_distance<Format_t,NDim>;
    rect_dist_fn  euclidean_rect_dist  =
            &kernels::euclidean_distance<false,Format_t,NDim>;
    map["euclidean_point_distance"].getFrom(euclidean_point_dist);
    map["euclidean_rect_distance"] .getFrom(euclidean_rect_dist);

    SE3FAttr<Format_t,NDim,(NDim>=7)>::append(map);
    SO3FAttr<Format_t,NDim,(NDim>=4)>::append(map);
    cuda::bitonic::get_fattr_kv<Format_t,Format_t>(map);
}





} //< namespace cudaNN
} //< namespace mpblocks


#endif // FATTR_CU_HPP_
