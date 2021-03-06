/*
 *  Copyright (C) 2012 Josh Bialkowski (jbialk@mit.edu)
 *
 *  This file is part of cpp-pthreads.
 *
 *  cpp-pthreads is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  cpp-pthreads is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with cpp-pthreads.  If not, see <http://www.gnu.org/licenses/>.
 */
/**
 *  @file   include/cpp-pthreads/enums.h
 *
 *  @date   Jan 3, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  Defines enumerators and constants
 */

#ifndef CPP_PTHREADS_ENUMS_H_
#define CPP_PTHREADS_ENUMS_H_

namespace pthreads {

/// controls whether a thread is created in a detached state, meaning that the
/// thread will destroy itself when it finished running, and cannot be joined
enum DetachState {
  DETACHED = 0,  ///< thread is started in a detached state, when thread
                 ///  terminates, memory is automatically freed
  JOINABLE = 1,  ///< thread must be join()ed by another thread for memory
                 ///  to be freed
  INVALID_DETACH_STATE
};

/// determines whether a thread inherits it's scheduling policy from the
/// creating thread or if it uses the policy stored in the attribute object
enum InheritSched {
  INHERIT = 0,   ///< thread inherits scheduling policy from creating
                 ///  thread
  EXPLICIT = 1,  ///< thread's scheduling policy is explicitly set in
                 ///  the Attr<Thread> object
  INVALID_INHERIT_SCHED
};

/// if multiple threads are waiting on the same mutex, the Scheduling Policy
/// determines the order in which they are aquired
enum SchedPolicy {
  OTHER,
  FIFO,
  RR,
  BATCH,
  IDLE,
  SPORADIC,
  INVALID_SCHED_POLICY
};

/// Signifies process or system scheduling contention scope
enum Scope {
  SYSTEM = 0,
  PROCESS = 1,
  INVALID_SCOPE
};

/// indicates whether or not a resource is shared across processes
enum PShared {
  SHARED,     ///< resource is shared by all processes who have access to
              ///  the memory containing the resource handle
  PRIVATE,    ///< resource is private to the creating process
  INVALID_PSHARED
};

/// Which progocol is used for priority
enum Protocol {
  PRIO_NONE,
  PRIO_INHERIT,
  PRIO_PROTECT,
  INVALID_PROTOCOL
};

/// Types of mutexes
enum Type {
  NORMAL,
  ERROR_CHECK,
  RECURSIVE,
  DEFAULT,
  INVALID_TYPE
};

/// returns pointer to a map from enums to pthread enums (ints)
template<typename Enum>
int* getEnumMap();

/// returns the pthread integer enumerator that corresponds to the
/// specified value
template<typename Enum>
inline int mapEnum(Enum val) {
  return getEnumMap<Enum>()[val];
}

/// returns the pthread integer enumerator that corresponds to the
/// specified value
template<typename Enum>
Enum getEnum(int val);

} // namespace pthreads














#endif // ENUMS_H_
