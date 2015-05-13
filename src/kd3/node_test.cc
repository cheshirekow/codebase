/*
 *  Copyright (C) 2012 Josh Bialkowski (josh.bialkowski@gmail.com)
 *
 *  This file is part of kd3.
 *
 *  kd3 is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  kd3 is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with kd3.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <gtest/gtest.h>
#include <kd3/node.h>


TEST(NodeTest, InsertTest) {
    kd3::Node<double, 2> root;
    kd3::Node<double, 2> left;
    kd3::Node<double, 2> right;

    root.SetPoint({0.5, 0.5});

    /* hrect scope */ {
      kd3::HyperRect<double, 2> hrect{{0, 0}, {1, 1}};
      left.SetPoint({0.25, 0.5});
      root.Insert(&hrect, &left);
      // since root was default constructed, it will split on the zero'th axis
      // and so the hyper-rectangle of the child should be the following
      EXPECT_EQ(0.0, hrect.min_ext[0]);
      EXPECT_EQ(0.0, hrect.min_ext[1]);
      EXPECT_EQ(0.5, hrect.max_ext[0]);
      EXPECT_EQ(1.0, hrect.max_ext[1]);
    }

    /* hrect scope */ {
      kd3::HyperRect<double, 2> hrect{{0, 0}, {1, 1}};
      right.SetPoint({0.75, 0.5});
      root.Insert(&hrect, &right);
      EXPECT_EQ(0.5, hrect.min_ext[0]);
      EXPECT_EQ(0.0, hrect.min_ext[1]);
      EXPECT_EQ(1.0, hrect.max_ext[0]);
      EXPECT_EQ(1.0, hrect.max_ext[1]);
    }

    EXPECT_EQ(&left, root.smaller_child());
    EXPECT_EQ(&right, root.greater_child());
}
