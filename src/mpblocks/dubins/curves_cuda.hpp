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
 *  \file   mpblocks/dubins/curves.h
 *
 *  \date   Oct 30, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#ifndef MPBLOCKS_DUBINS_CURVES_CUDA2_HPP_
#define MPBLOCKS_DUBINS_CURVES_CUDA2_HPP_


#include <mpblocks/cuda.hpp>
#include <mpblocks/cuda/linalg2.h>

#include <mpblocks/dubins/curves_cuda.h>
#include <mpblocks/dubins/curves/funcs.hpp>
#include <mpblocks/dubins/curves_cuda/impl/SolutionLRLa.hpp>
#include <mpblocks/dubins/curves_cuda/impl/SolutionLRLb.hpp>
#include <mpblocks/dubins/curves_cuda/impl/SolutionRLRa.hpp>
#include <mpblocks/dubins/curves_cuda/impl/SolutionRLRb.hpp>
#include <mpblocks/dubins/curves_cuda/impl/SolutionLSL.hpp>
#include <mpblocks/dubins/curves_cuda/impl/SolutionRSR.hpp>
#include <mpblocks/dubins/curves_cuda/impl/SolutionLSR.hpp>
#include <mpblocks/dubins/curves_cuda/impl/SolutionRSL.hpp>
#include <mpblocks/dubins/curves_cuda/PackedIndex.hpp>


#endif // CURVES_H_
