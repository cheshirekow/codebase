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


#include <mpblocks/gtk.hpp>
#include "cuda_helper.h"

namespace mpblocks {
namespace examples {
namespace   dubins {

class CudaHelper_impl : public CudaHelper {
 public:
  virtual ~CudaHelper_impl(){};
  virtual void populateDevices(gtk::LayoutMap& layout);
};

CudaHelper* create_cudaHelper() { return new CudaHelper_impl(); }

void CudaHelper_impl::populateDevices(gtk::LayoutMap& layout) {
  Gtk::ComboBoxText* combo = layout.widget<Gtk::ComboBoxText>("cudaDevices");
  combo->append("No devices, compiled without cuda");
  combo->set_active(0);
}

}  // dubins
}  // examples
}  // mpblocks
