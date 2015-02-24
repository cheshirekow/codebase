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
 *  @date   Nov 5, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief
 */

#ifndef MPBLOCKS_DUBINS_CURVES_EIGEN_TRIG_WRAP_H_
#define MPBLOCKS_DUBINS_CURVES_EIGEN_TRIG_WRAP_H_

namespace     mpblocks {
namespace       dubins {
namespace curves_eigen {

/// provides static functions which call the appropriate trig function
/// based in the template parameter specifying the path Primitive;
/// @see specializations in trig_wrap.hpp
template <typename Format_t, class Primitive>
struct TrigWrap{};

} // curves_eigen
} // dubins
} // mpblocks

#endif // MPBLOCKS_DUBINS_CURVES_EIGEN_TRIG_WRAP_H_
