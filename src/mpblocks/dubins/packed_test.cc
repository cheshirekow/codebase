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
 *  \file   src/main.cpp
 *
 *  \date   Oct 24, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */


#include <iostream>
#include <iomanip>
#include <bitset>

#include <gtest/gtest.h>
#include <mpblocks/dubins/curves_cuda/PackedIndex.hpp>

using namespace mpblocks;

TEST(PackedTypeTest, PoorlyNamedTest) {
  typedef std::bitset<32> bits;
  typedef dubins::curves_cuda::PackedIndex<float> packed;

  std::cout << "      0  : " << bits((float)0) << "\n";
  std::cout << " (INV,0) : " << bits(packed().getUnsigned()) << "\n";
  packed p;
  p.setId(dubins::LRLb);
  p.setIdx(0x0F0);

  uint32_t packed_bits  = reinterpret_cast<uint32_t&>(p);
  float    packed_float = p.getPun();
  uint32_t float_bits   = reinterpret_cast<uint32_t&>(packed_float);

  EXPECT_EQ(packed_bits, float_bits);
  EXPECT_EQ(packed_bits, p.getUnsigned());
  std::cout << "   (0,1) : " << bits(packed_bits) << "\n";
  std::cout << "   (0,1) : " << bits(float_bits) << "\n";
  std::cout << "   (0,1) : " << bits(p.getUnsigned()) << "\n";
}
