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
 *  @file   /home/josh/Codes/cpp/mpblocks2/robot_nn/src/r2s1/impl/CudaImplementation.cpp
 *
 *  @date   Nov 7, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */



#include <Eigen/Dense>
#include <mpblocks/cudaNN/PointSet.h>

#include "config.h"
#include "Implementation.h"
#include "impl/ctors.h"


namespace mpblocks {
namespace robot_nn {



struct CudaImplementation:
    public Implementation
{
    cudaNN::PointSet<float,3>    cuda_points;
    cudaNN::ResultBlock<float,2> cuda_result;

    virtual ~CudaImplementation(){}

    virtual void allocate( int maxSize )
    {
        cuda_points.allocate(maxSize);
        cuda_result.allocate(k);
    }

    virtual void insert_derived( int i, const Point& x )
    {
        cuda_points.insert( x.data() );
    }

    virtual void findNearest_derived( const Point& q )
    {
        cuda_points.nearest(cudaNN::R2S1(w),q.data(),size,cuda_result);
    }

    virtual void get_result( std::vector<int>& out )
    {
        out.resize(k);
        for(int i=0; i < k; i++)
            out[i] = cuda_result(1,i);
    }
};


Implementation* impl_cuda()
{
    return new CudaImplementation();
}


}
}



