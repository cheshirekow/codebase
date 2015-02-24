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
 *  @file   /home/josh/Codes/cpp/mpblocks2/robot_nn/src/r2s1/impl/ctors.h
 *
 *  @date   Nov 7, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_ROBOT_NN_R2S1_IMPL_CTORS_H_
#define MPBLOCKS_ROBOT_NN_R2S1_IMPL_CTORS_H_

#include "config.h"
#include "Implementation.h"


namespace mpblocks {
namespace robot_nn {
namespace      se3 {

typedef Eigen::Matrix<float,7,1> Point;

Implementation* impl_gpu_bf ( );
Implementation* impl_cpu_bf ( std::vector<Point>& points );
Implementation* impl_cpu_sqt( std::vector<Point>& points, int capacity, int rate );
Implementation* impl_gpu_sqt( std::vector<Point>& points, int capacity, int rate );
Implementation* impl_cpu_rqt( std::vector<Point>& points, int capacity );
Implementation* impl_gpu_rqt( std::vector<Point>& points, int capacity );

}
}
}




#endif // CTORS_H_
