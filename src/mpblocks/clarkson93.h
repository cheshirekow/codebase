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
 *  @date   Aug 17, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 */

#ifndef MPBLOCKS_CONVEX_HULL_CLARKSON93_H_
#define MPBLOCKS_CONVEX_HULL_CLARKSON93_H_


#include <cassert>
#include <iostream>
#include <list>
#include <set>
#include <vector>

#include <Eigen/Dense>


namespace    mpblocks {


/// Implementation of RIC convex-hull construction (but not maintainance)
/// algorithm by Clarkson (Computational Geometry, '93)
/**
 *  The algorithm builds a data structure representing the convex hull by
 *  Random Incremental Construction. An (extended) Triangulation of inserted
 *  points is maintained. The triangulation is extended in such a way that
 *  every point in @f$ \mathbb{R}^d @f$ is contained in some simplex.
 *  Specifically, every facet of the convex hull is in fact a member of two
 *  simplices. One, which is a normal simplex, is contained within the
 *  convex hull. The other lies outside the convex hull, and it's peak vertex
 *  is the so called 'anti-origin' (or vertex 'at infinity in CGAL speak).
 *
 *  @see http://www.sciencedirect.com/science/article/pii/092577219390009U
 */
namespace  clarkson93 {


const int Dynamic = -1;

enum Orientation{ INSIDE, OUTSIDE};

struct WithoutSet{};

// forward declarations of templates
template <class Traits>
struct HorizonRidge;

template <class Traits>
struct SimplexBase;

template <class Traits>
struct DefaultSimplex;

template <class Traits>
struct Triangulation;

} // namespace clarkson93
} // namespace mpblocks


#include <mpblocks/clarkson93/BitMember.h>
#include <mpblocks/clarkson93/Indexed.h>
#include <mpblocks/clarkson93/PQueue.h>
#include <mpblocks/clarkson93/StaticStack.h>
#include <mpblocks/clarkson93/HorizonRidge.h>
#include <mpblocks/clarkson93/Simplex.h>
#include <mpblocks/clarkson93/Simplex2.h>
#include <mpblocks/clarkson93/Triangulation.h>
#include <mpblocks/clarkson93/ExampleTraits.h>
#include <mpblocks/clarkson93/ExampleTraits2.h>

#include <mpblocks/clarkson93/dynamic/Simplex.h>
#include <mpblocks/clarkson93/dynamic/Triangulation.h>


#endif // CLARKSON_H_
