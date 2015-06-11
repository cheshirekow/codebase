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
#ifndef CLARKSON93_HORIZON_RIDGE_H_
#define CLARKSON93_HORIZON_RIDGE_H_

#include <mpblocks/clarkson93.h>

namespace clarkson93 {

/// A horizon ridge is a d-2 dimensional facet (i.e. a facet of a facet),
/**
 *  We store a representation of a horizon ridge by a pair of simplices, one
 *  of which is x-visible and the other of which is not
 */
template <class Traits>
struct HorizonRidge {
  typedef typename Traits::SimplexRef SimplexRef;

  SimplexRef Svis;    ///< the x-visible simplex, which is dropped to a
                      ///  finite one
  SimplexRef Sinvis;  ///< the x-invisible simplex, which remains infinite
  SimplexRef Sfill;   ///< the new simplex created to fill the wedge
                      ///  vacated by Svis

  HorizonRidge(SimplexRef Svis_in, SimplexRef Sinvis_in)
      : Svis(Svis_in), Sinvis(Sinvis_in) {}
};

}  // namespace clarkson93

#endif  // CLARKSON93_HORIZON_RIDGE_H_
