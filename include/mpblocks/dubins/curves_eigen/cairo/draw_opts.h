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
 *  @file
 *  @date   Oct 30, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief
 */

#ifndef MPBLOCKS_DUBINS_CURVES_EIGEN_DRAWOPTS_H_
#define MPBLOCKS_DUBINS_CURVES_EIGEN_DRAWOPTS_H_

#include <bitset>

namespace     mpblocks {
namespace       dubins {
namespace curves_eigen {

/// encapsulates patterns for drawing
struct DrawOpts {
  Cairo::RefPtr<Cairo::Pattern> patL;
  Cairo::RefPtr<Cairo::Pattern> patR;
  Cairo::RefPtr<Cairo::Pattern> patS;
  Cairo::RefPtr<Cairo::Pattern> patPath;
  Cairo::RefPtr<Cairo::Context> ctx;

  bool extra;
  bool useDashExtra;
  bool useDashPath;
  double dashOffExtra;
  double dashOffPath;

  bool drawBalls;           ///< whether or not to draw balls
  unsigned int whichSpec;   ///< which constraint spec to draw
  std::bitset<8> solnBits;  ///< if not bits are set, then draw the min
                            ///  solution
  std::bitset<6> drawBits;  ///< which variants to draw of the solution
                            ///  (6 bit means draw the min and no others)

  std::vector<double> dash;

  DrawOpts() : dash(2) {
    drawBalls = true;
    extra = true;
    useDashExtra = true;
    useDashPath = false;
    dash[0] = 0.1;
    dash[1] = 0.1;
    dashOffExtra = 0;
    dashOffPath = 0.1;
  }
};

} // curves_eigen
} // dubins
} // mpblocks




#endif // QUERY_H_
