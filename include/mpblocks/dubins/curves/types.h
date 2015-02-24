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
 *
 *  @date   Nov 4, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief
 */

#ifndef MPBLOCKS_DUBINS_CURVES_TYPES_H_
#define MPBLOCKS_DUBINS_CURVES_TYPES_H_

namespace mpblocks {
namespace   dubins {

/// empty struct used to template "left turn" primitive
struct L{};

/// empty struct used to template "right turn" primitive
struct R{};

/// empty struct used to template "straight" primitive
struct S{};

/// empty struct used to template "variant" of three arc primitives
struct a{};

/// empty struct used to template "variant" of three arc primitives
struct b{};

/// enumerates solution types
enum SolutionId
{
    LRLa,
    LRLb,
    RLRa,
    RLRb,
    LSL,
    RSR,
    LSR,
    RSL,
    INVALID
};

/// converts template paramters to SolutionID
template <class P1, class P2, class P3, class V=a>
struct TypeToId{ static const SolutionId ID = INVALID; };

// these types have a variant
template <> struct TypeToId<L,R,L,a>{ static const SolutionId ID = LRLa; };
template <> struct TypeToId<L,R,L,b>{ static const SolutionId ID = LRLb; };
template <> struct TypeToId<R,L,R,a>{ static const SolutionId ID = RLRa; };
template <> struct TypeToId<R,L,R,b>{ static const SolutionId ID = RLRb; };

// these types dont have a variant
template <class V> struct TypeToId<L,S,L,V>{ static const SolutionId ID = LSL; };
template <class V> struct TypeToId<R,S,R,V>{ static const SolutionId ID = RSR; };
template <class V> struct TypeToId<L,S,R,V>{ static const SolutionId ID = LSR; };
template <class V> struct TypeToId<R,S,L,V>{ static const SolutionId ID = RSL; };

/// converts a SolutionID and index 0,1,2 into a type
template <SolutionId ID, int i>
struct IdToType{ typedef void Result; };

template <> struct IdToType<LRLa,0>{ typedef L Result; };
template <> struct IdToType<LRLa,1>{ typedef R Result; };
template <> struct IdToType<LRLa,2>{ typedef L Result; };
template <> struct IdToType<LRLb,0>{ typedef L Result; };
template <> struct IdToType<LRLb,1>{ typedef R Result; };
template <> struct IdToType<LRLb,2>{ typedef L Result; };

template <> struct IdToType<RLRa,0>{ typedef R Result; };
template <> struct IdToType<RLRa,1>{ typedef L Result; };
template <> struct IdToType<RLRa,2>{ typedef R Result; };
template <> struct IdToType<RLRb,0>{ typedef R Result; };
template <> struct IdToType<RLRb,1>{ typedef L Result; };
template <> struct IdToType<RLRb,2>{ typedef R Result; };

template <> struct IdToType<LSL,0>{ typedef L Result; };
template <> struct IdToType<LSL,1>{ typedef S Result; };
template <> struct IdToType<LSL,2>{ typedef L Result; };
template <> struct IdToType<LSR,0>{ typedef L Result; };
template <> struct IdToType<LSR,1>{ typedef S Result; };
template <> struct IdToType<LSR,2>{ typedef R Result; };

template <> struct IdToType<RSR,0>{ typedef R Result; };
template <> struct IdToType<RSR,1>{ typedef S Result; };
template <> struct IdToType<RSR,2>{ typedef R Result; };
template <> struct IdToType<RSL,0>{ typedef R Result; };
template <> struct IdToType<RSL,1>{ typedef S Result; };
template <> struct IdToType<RSL,2>{ typedef L Result; };

} // dubins
} // mpblocks

#endif  // MPBLOCKS_DUBINS_CURVES_TYPES_H_
