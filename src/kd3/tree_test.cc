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
#include <kd3/tree.h>
#include <kd3/enumerator.h>

TEST(TreeTest, EnumerateTest) {
  typedef kd3::Node<double, 2> Node;
  kd3::Tree<double,2> tree({{0.0, 0.0}, {1.0, 0.001}});
  tree.Insert(new Node({0.5, 0.0}));
  tree.Insert(new Node({0.25, 0.0}));
  tree.Insert(new Node({0.75, 0.0}));
  tree.Insert(new Node({0.1, 0.0}));
  tree.Insert(new Node({0.3, 0.0}));

  auto list = tree.Enumerate();
  ASSERT_EQ(5, list.size());
}
