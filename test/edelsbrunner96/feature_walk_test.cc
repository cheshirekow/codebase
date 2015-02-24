/*
 *  Copyright (C) 2014 Josh Bialkowski (jbialk@mit.edu)
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
 *  @date   Sept 22, 2014
 *  @author Josh Bialkowski (josh.bialkowski@gmail.com)
 *  @brief
 */

#include <gtest/gtest.h>
#include <edelsbrunner96.hpp>
#include "char_label_triangulation.h"


/// Given a container (i.e. list, vector) of chars, format them into a string
template <class Container>
std::string FormatLink(const Container& container) {
  if(container.size() < 1) {
    return "[ ]";
  }
  std::stringstream strm;
  strm << "[ ";
  auto last = container.end();
  --last;
  for(auto iter = container.begin(); iter != last; iter++) {
    strm << *iter << ", ";
  }
  strm << *last << " ]";
  return strm.str();
}

/// A manually built triangulation for feature walk tests
/**
 *  Ascii art for this triangulation
 *  @verbataim
 *  lowercase : vertices
 *  uppercase : simplices
 *
 *       \     |     |    /         \     |     |    /
 *        \    |     |   /           \  X | Y   | Z /
 *        k\__l|____m|__/n            \___|_____|__/
 *         |  /\    /\  |             |R /\ T  /\ V|
 *         | /  \  /  \ |         Q   | /  \  /  \ |  W
 *    ____h|/____\/i___\|j___    _____|/_S__\/_U__\|____
 *         |\    /\    /|             |\ L  /\ N  /|
 *         | \  /  \  / |         J   | \  /  \  / |  P
 *    ____d|__\/e___\/f_|g___    _____|K_\/_M__\/_O|____
 *         |  /\    /\  |             |D /\ F  /\ H|
 *         | /  \  /  \ |         C   | /  \  /  \ |  I
 *         |/____\/____\|             |/_E__\/__G_\|
 *        a/     |b     \c            /     |      \
 *        /      |       \           /  A   |   B   \
 *       /       |        \         /       |        \
 *
@endverbatim
 */
class FeatureWalkTest : public CharLabelTriangulationTest {
 public:
  virtual void SetUp() {
    typedef Traits::Scalar Scalar;
    typedef Traits::Simplex Simplex;
    typedef Traits::SimplexRef SimplexRef;
    typedef Traits::Point Point;
    typedef Traits::PointRef PointRef;

    // fill the vertex storage
    storage_.points['a'] = Point(0,0);
    storage_.points['b'] = Point(2,0);
    storage_.points['c'] = Point(4,0);
    storage_.points['d'] = Point(0,1);
    storage_.points['e'] = Point(1,1);
    storage_.points['f'] = Point(3,1);
    storage_.points['g'] = Point(4,1);
    storage_.points['h'] = Point(0,2);
    storage_.points['i'] = Point(2,2);
    storage_.points['j'] = Point(4,2);
    storage_.points['k'] = Point(0,3);
    storage_.points['l'] = Point(1,3);
    storage_.points['m'] = Point(3,3);
    storage_.points['n'] = Point(4,3);

    // setup simplices and neighbor lists
    SetupSimplex('A', {'\0', 'a', 'b'}, {'E', 'B', 'C'});
    SetupSimplex('B', {'\0', 'b', 'c'}, {'G', 'I', 'A'});
    SetupSimplex('C', {'\0', 'a', 'd'}, {'D', 'J', 'A'});
    SetupSimplex('D', {'a', 'd', 'e'}, {'K', 'E', 'C'});
    SetupSimplex('E', {'a', 'b', 'e'}, {'F', 'D', 'A'});
    SetupSimplex('F', {'b', 'e', 'f'}, {'M', 'G', 'E'});
    SetupSimplex('G', {'b', 'c', 'f'}, {'H', 'F', 'B'});
    SetupSimplex('H', {'c', 'f', 'g'}, {'O', 'I', 'G'});
    SetupSimplex('I', {'\0', 'c', 'g'}, {'H', 'P', 'B'});
    SetupSimplex('J', {'\0', 'd', 'h'}, {'K', 'Q', 'C'});
    SetupSimplex('K', {'d', 'e', 'h'}, {'L', 'J', 'D'});
    SetupSimplex('L', {'e', 'h', 'i'}, {'S', 'M', 'K'});
    SetupSimplex('M', {'e', 'f', 'i'}, {'N', 'L', 'F'});
    SetupSimplex('N', {'f', 'i', 'j'}, {'U', 'O', 'M'});
    SetupSimplex('O', {'f', 'g', 'j'}, {'P', 'N', 'H'});
    SetupSimplex('P', {'\0', 'g', 'j'}, {'O', 'W', 'I'});
    SetupSimplex('Q', {'\0', 'h', 'k'}, {'R', 'X', 'J'});
    SetupSimplex('R', {'h', 'k', 'l'}, {'X', 'S', 'Q'});
    SetupSimplex('S', {'h', 'i', 'l'}, {'T', 'R', 'L'});
    SetupSimplex('T', {'i', 'l', 'm'}, {'Y', 'U', 'S'});
    SetupSimplex('U', {'i', 'j', 'm'}, {'V', 'T', 'N'});
    SetupSimplex('V', {'j', 'm', 'n'}, {'Z', 'W', 'U'});
    SetupSimplex('W', {'\0', 'j', 'n'}, {'V', 'Z', 'P'});
    SetupSimplex('X', {'\0', 'k', 'l'}, {'R', 'Y', 'Q'});
    SetupSimplex('Y', {'\0', 'l', 'm'}, {'T', 'Z', 'X'});
    SetupSimplex('Z', {'\0', 'm', 'n'}, {'V', 'W', 'Y'});
  }

  void Do(Traits::SimplexRef start,
          std::initializer_list<Traits::PointRef> feature,
          std::initializer_list<Traits::SimplexRef> expected_link) {
    ASSERT_PRED_FORMAT1(AssertGraphConsistentFrom, 'A');

    typedef Traits::Scalar Scalar;
    typedef Traits::Simplex Simplex;
    typedef Traits::SimplexRef SimplexRef;
    typedef Traits::Point Point;
    typedef Traits::PointRef PointRef;

    std::vector<Traits::PointRef> feature_;
    std::vector<Traits::SimplexRef> expected_link_;
    std::vector<Traits::SimplexRef> actual_link_;

    std::copy(feature.begin(), feature.end(), std::back_inserter(feature_));
    std::copy(expected_link.begin(), expected_link.end(),
              std::back_inserter(expected_link_));
    std::sort(feature_.begin(), feature_.end());
    std::sort(expected_link_.begin(), expected_link_.end());
    edelsbrunner96::FeatureWalk<Traits>(storage_, start, feature_.begin(),
                                        feature_.end(),
                                        std::back_inserter(actual_link_));
    std::sort(actual_link_.begin(), actual_link_.end());

    EXPECT_EQ(actual_link_, expected_link_)
        << "   Feature: " << FormatLink(feature_) << "\n   Start at: " << start
        << "\n   Actual link: " << FormatLink(actual_link_)
        << "\n   Expected link: " << FormatLink(expected_link_);
  }
};


// Point/vertex features
TEST_F(FeatureWalkTest, De) { Do('D', {'e'}, {'D', 'E', 'F', 'K', 'L', 'M'}); }
TEST_F(FeatureWalkTest, Ee) { Do('E', {'e'}, {'D', 'E', 'F', 'K', 'L', 'M'}); }
TEST_F(FeatureWalkTest, Ke) { Do('K', {'e'}, {'D', 'E', 'F', 'K', 'L', 'M'}); }
TEST_F(FeatureWalkTest, Li) { Do('L', {'i'}, {'L', 'M', 'N', 'S', 'T', 'U'}); }
TEST_F(FeatureWalkTest, Ni) { Do('N', {'i'}, {'L', 'M', 'N', 'S', 'T', 'U'}); }
TEST_F(FeatureWalkTest, Ti) { Do('T', {'i'}, {'L', 'M', 'N', 'S', 'T', 'U'}); }
// line-segment/edge features
TEST_F(FeatureWalkTest, Fef) { Do('F', {'e', 'f'}, {'F', 'M'}); }
TEST_F(FeatureWalkTest, Mef) { Do('M', {'e', 'f'}, {'F', 'M'}); }
TEST_F(FeatureWalkTest, Til) { Do('T', {'i','l'}, {'S', 'T'}); }
// triangle/simplex features
TEST_F(FeatureWalkTest, Tilm) { Do('T', {'i','l','m'}, {'T'}); }


