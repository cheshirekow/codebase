/*
 *  Copyright (C) 2012 Josh Bialkowski (josh.bialkowski@gmail.com)
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
#ifndef CLARKSON93_BIT_MEMBER_H_
#define CLARKSON93_BIT_MEMBER_H_

#include <bitset>

namespace clarkson93 {

/// dummy class which allows us to use SNFINAE
struct BitMemberBase {};

/// indicates membership into a number of sets by a bitfield
/**
 *  There are several interesting sets that a particular simplex in the
 *  triangulation may be a member of, particularly during certain walks and
 *  searches. In several cases it is useful to indicate set membership within
 *  the object itself. This class provides a thin wrapper around std::bitset
 *  in order to reference individual set-membership bits by an index from an
 *  enumeration.
 *
 *  For example:
 *  @code
    enum MySets {
      SET_A = 0,
      SET_B,
      SET_C,
      NUM_SETS
    };

    BitMember<MySets,NUM_SETS> set_member;
    set_member.AddTo(SET_A);
    set_member.AddTo(SET_B);
    bool is_member_of_a = set_member.IsMemberOf(SET_A); // true
    bool is_member_of_b = set_member.IsMemberOf(SET_B); // true
    bool is_member_of_c = set_member.IsMemberOf(SET_C); // false
@endcode
 */
template <typename Enum, unsigned int size_>
struct BitMember : public BitMemberBase, public std::bitset<size_> {
  /// mark this object as a member of the @set_bit set
  void AddTo(Enum set_bit) {
    (*this)[set_bit] = true;
  }

  /// mark this object as not-a-member of the @set_bit set
  void RemoveFrom(Enum set_bit) {
    (*this)[set_bit] = false;
  }

  /// return true if this object is a member of the @set_bit set
  bool IsMemberOf(Enum set_bit) const {
    return (*this)[set_bit];
  }
};

}  // namespace clarkson93

#endif  // CLARKSON93_BIT_MEMBER_H_
