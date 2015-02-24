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
 *  \file   src/CudaHelper.h
 *
 *  \date   Oct 24, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#ifndef MPBLOCKS_EXAMPLES_DUBINS_CURVES_CUDA_HELPER_
#define MPBLOCKS_EXAMPLES_DUBINS_CURVES_CUDA_HELPER_

#include <gtkmm.h>
#include <mpblocks/gtk.h>
#include <Eigen/Dense>

namespace mpblocks {
namespace examples {
namespace   dubins {

/// interface for cuda helper
class CudaHelper {
 public:
  typedef Eigen::Matrix<float, 3, 1> Vector3f;
  typedef Eigen::Matrix<double, 3, 1> Vector3d;

 public:
  virtual ~CudaHelper() {}

  virtual void populateDevices(gtk::LayoutMap& layout) = 0;
  virtual void populateDetails(gtk::LayoutMap& layout) {}

  virtual void solve(gtk::LayoutMap& layout, Vector3d& q0, Vector3d& q1,
                     double r) {}
};

CudaHelper* create_cudaHelper();

} // dubins
} // examples
} // mpblocks


#endif // CUDA_HELPER_
