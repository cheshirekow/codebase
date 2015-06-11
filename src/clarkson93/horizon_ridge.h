/*
 *  Copyright (C) 2012 Josh Bialkowski (josh.bialkowski@gmail.com)
 *
 *  This file is part of clarkson93.
 *
 *  clarkson93 is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  clarkson93 is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with clarkson93.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef CLARKSON93_HORIZON_RIDGE_H_
#define CLARKSON93_HORIZON_RIDGE_H_

#include <clarkson93/simplex.h>

namespace clarkson93 {

/// A horizon ridge is a d-2 dimensional facet (i.e. a facet of a facet),
/**
 *  We store a representation of a horizon ridge by a pair of simplices, one
 *  of which is x-visible and the other of which is not
 */
template <class Traits>
struct HorizonRidge {
  Simplex<Traits>* x_visible;    ///< the x-visible simplex, which is dropped to
                                 /// a finite one
  Simplex<Traits>* x_invisible;  ///< the x-invisible simplex, which remains
                                 /// infinite
  Simplex<Traits>* fill;         ///< the new simplex created to fill the wedge
                                 ///  vacated by Svis

  HorizonRidge(Simplex<Traits>* x_visible_in, Simplex<Traits>* x_invisible_in)
      : x_visible(x_visible_in), x_invisible(x_invisible_in), fill(nullptr) {}
};

}  // namespace clarkson93

#endif  // CLARKSON93_HORIZON_RIDGE_H_
