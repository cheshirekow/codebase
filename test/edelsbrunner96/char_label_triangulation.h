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
 *  @date   Oct 25, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief
 */

#ifndef TEST_EDELSBRUNNER96_CHAR_LABEL_TRIANGULATION_H_
#define TEST_EDELSBRUNNER96_CHAR_LABEL_TRIANGULATION_H_

#include <map>
#include <boost/format.hpp>
#include <gtest/gtest.h>
#include <edelsbrunner96.hpp>

/// Given a container (i.e. list, vector) of chars, format them into a string
template <class Container>
std::string FormatSet(const Container& container) {
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

template <typename T1, typename T2>
::testing::AssertionResult AssertCharSetEquals(
    const std::string& set1_expr,
    const std::string& set2_expr,
    const T1& set1_in, const T2& set2_in) {
  std::vector<char> set1(set1_in.begin(), set1_in.end());
  std::vector<char> set2(set2_in.begin(), set2_in.end());
  std::sort(set1.begin(), set1.end());
  std::sort(set2.begin(), set2.end());

  if (set1 == set2) {
    return ::testing::AssertionSuccess();
  } else {
    int len = std::max(set1_expr.size(), set2_expr.size());
    std::stringstream fmt_strm;
    fmt_strm << "%" << len << "s";
    return ::testing::AssertionFailure()
           << set1_expr << " != " << set2_expr << "\n where "
           << boost::format(fmt_strm.str()) % set1_expr
           << " = " << FormatSet(set1) << "\n   and "
           << boost::format(fmt_strm.str()) % set2_expr
           << " = " << FormatSet(set2);
  }
}

template <typename T>
::testing::AssertionResult AssertCharSetOrdered(
    const std::string& set_expr,
    const T& set_in) {
  std::vector<char> set;
  set.reserve(set_in.size());
  std::copy(set_in.begin(), set_in.end(), std::back_inserter(set));

  for(int i=0; i < set.size()-1; i++) {
    if(set[i] > set[i+1]) {
      return ::testing::AssertionFailure()
        << set_expr << " is not ordered"
        << "\n where " << set_expr << " = " << FormatSet(set);
    }
  }
  return ::testing::AssertionSuccess();
}



/// Act's just like a char but has a distinct type so that type-deduction works
/// between SimplexRef and PointRef
template <int i>
struct Index {
  char value;
  Index() : value('\0') {}
  Index(char v) : value(v) {}
  operator char&() { return value; }
  operator const char&() const { return value; }
};

/// A traits class for manually built triangulations where point references and
/// simplex references are char's, making it easier to understand labels.
struct Traits {
  static const int NDim = 2;

  typedef double Scalar;
  typedef Index<0> SimplexRef;
  typedef Index<1> PointRef;

  typedef Eigen::Matrix<Scalar, NDim, 1> Point;
  typedef edelsbrunner96::SimplexBase<Traits> Simplex;

  class Storage {
   public:
    std::map<char, Point> points;
    std::map<char, Simplex> simplices;

    const Point& operator[](const PointRef p) { return points[p.value]; }

    PointRef NullPoint() { return '\0'; }

    Simplex& operator[](const SimplexRef s) { return simplices[s.value]; }

    SimplexRef NullSimplex() { return '\0'; }

    // for testing, there is no simplex reuse
    void Retire(SimplexRef s_ref) {
      simplices[s_ref].version++;
    }

    // for testing, there is no simplex reuse
    SimplexRef Promote() {
      for (char label = 'A'; label < 'Z'; label++) {
        if (simplices.count(label) > 0) {
          continue;
        } else {
          simplices[label].version = 0;
          return label;
        }
      }
      assert(false);
      return 0;
    }

    bool Contains(const PointRef p) {
      if (p == NullPoint()) {
        return true;
      } else {
        return points.count(p) > 0;
      }
    }

    bool Contains(const SimplexRef s) {
      if (s == NullSimplex()) {
        return true;
      } else {
        return simplices.count(s) > 0;
      }
    }
  };
};

/// Simplifies the process of creating a manual triangulation
class CharLabelTriangulationTest : public testing::Test {
 public:
  typedef Traits::SimplexRef SimplexRef;
  typedef Traits::Simplex Simplex;
  typedef Traits::PointRef PointRef;

  void SetupSimplex(char simplex_label,
                    std::initializer_list<char> vertex_labels,
                    std::initializer_list<char> neighbor_labels) {
    Traits::Simplex& s = storage_.simplices[simplex_label];
    std::copy(vertex_labels.begin(), vertex_labels.end(), s.V.begin());
    std::copy(neighbor_labels.begin(), neighbor_labels.end(), s.N.begin());
  }

  ::testing::AssertionResult AssertGraphConsistentFrom(
      const std::string& start_label,
      const SimplexRef start_ref) {

    for (SimplexRef s_ref :
         edelsbrunner96::BreadthFirst<Traits>(storage_, start_ref)) {
      if (storage_.simplices.find(s_ref) == storage_.simplices.end()) {
        return ::testing::AssertionFailure()
               << "BreadthFirst reached a simplex " << s_ref
               << " which is not foundin storage\n";
      }

      Simplex& s = storage_.simplices[s_ref];
      std::string vertex_set_label =
          boost::str(boost::format("simplex[%s].V") % s_ref);
      ::testing::AssertionResult ordered_result =
          AssertCharSetOrdered(vertex_set_label, s.V);
      if (!ordered_result) {
        return ordered_result;
      }

      for (PointRef v : s.V) {
        if (storage_.points.find(v) == storage_.points.end() &&
            v != storage_.NullPoint()) {
          return ::testing::AssertionFailure()
                 << "Simplex " << s_ref << " contains a vertex " << v
                 << " which is not in the storage ";
        }
        SimplexRef n_ref = s.NeighborAcross(v);
        if (storage_.simplices.find(n_ref) == storage_.simplices.end()) {
          return ::testing::AssertionFailure()
                 << "Simplex " << s_ref << " contains a neighbor " << n_ref
                 << " which is not in the storage ";
        }
        Simplex& n = storage_[n_ref];

        std::string vertex_set_label =
            boost::str(boost::format("simplex[%s].V") % n_ref);
        ::testing::AssertionResult ordered_result =
            AssertCharSetOrdered(vertex_set_label, s.V);
        if (!ordered_result) {
          return ordered_result;
        }

        std::vector<PointRef> v_1, v_2;
        set::SymmetricDifference(s.V.begin(), s.V.end(), n.V.begin(), n.V.end(),
                                 std::back_inserter(v_1),
                                 std::back_inserter(v_2));
        if (v_1.size() != 1 || v_2.size() != 1 || v_1[0] != v) {
          return ::testing::AssertionFailure()
                 << "Inconsistent vertices for neighbor pair where:"
                 << "\n s_ref: " << s_ref
                 << "\n s.V: " << FormatSet(s.V)
                 << "\n n_ref: " << n_ref
                 << "\n n.V: " << FormatSet(n.V)
                 << "\n";
        }

        SimplexRef self_ref = n.NeighborAcross(v_2[0]);
        if (self_ref != s_ref) {
          return ::testing::AssertionFailure()
                 << "Neighbor " << n_ref << " of " << s_ref
                 << " does not point back to " << s_ref << " where "
                 << "\n s.V: " << FormatSet(s.V)
                 << "\n s.N = " << FormatSet(s.N)
                 << "\n n.V = " << FormatSet(n.V)
                 << "\n n.N = " << FormatSet(n.N);
        }
      }
    }

    return ::testing::AssertionSuccess();
  }

  ::testing::AssertionResult AssertMarksCleared() {
    for (auto&pair : storage_.simplices) {
      if (0ul != pair.second.marked.to_ulong()) {
        return ::testing::AssertionFailure() << "Marks remain for simplex "
                                             << pair.first;
      }
    }
    return ::testing::AssertionSuccess();
  }

  template <typename Container>
  ::testing::AssertionResult AssertNeighborhoodIs(
      const std::string& s_ref_label,
      const std::string& expected_label,
      SimplexRef s_ref, const Container& expected_in) {
    if(storage_.simplices.find(s_ref) == storage_.simplices.end()) {
      return ::testing::AssertionFailure() << "Simplex " << s_ref
                                           << " does not exist";
    }
    std::vector<SimplexRef> expected(expected_in.begin(), expected_in.end());
    std::vector<SimplexRef> actual(storage_[s_ref].N.begin(),
                                   storage_[s_ref].N.end());
    return AssertCharSetEquals(expected_label, s_ref_label, expected, actual);
  }

 protected:
  Traits::Storage storage_;
};

#endif //  TEST_EDELSBRUNNER96_CHAR_LABEL_TRIANGULATION_H_
