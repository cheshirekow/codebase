/*
 *  Copyright (C) 2015 Josh Bialkowski (josh.bialkowski@gmail.com)
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
#include <gtest/gtest.h>
#include "clarkson93/zip.h"

TEST(ZipTest, SimpleTest) {
  std::vector<int> a{1, 2, 3};
  std::vector<int> b{4, 5, 6};

  std::vector<int> a_copy;
  std::vector<int> b_copy;

  int i = 0;
  for (auto tuple : clarkson93::Zip(a, b)) {
    EXPECT_EQ(3, std::get<1>(tuple) - std::get<0>(tuple)) << " for index "
                                                          << i++;
    a_copy.push_back(std::get<0>(tuple));
    b_copy.push_back(std::get<1>(tuple));
  }

  EXPECT_EQ(a, a_copy);
  EXPECT_EQ(b, b_copy);
}
