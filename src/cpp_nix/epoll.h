/*
 *  Copyright (C) 2012 Josh Bialkowski (josh.bialkowski@gmail.com)
 *
 *  This file is part of cpp-nix.
 *
 *  cpp-nix is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  cpp-nix is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with cpp-nix.  If not, see <http://www.gnu.org/licenses/>.
 */
/**
 *  @file
 *  @date Jun 22, 2014
 *  @author Josh Bialkowski
 */

#ifndef CPP_NIX_EPOLL_H_
#define CPP_NIX_EPOLL_H_

#include <sys/epoll.h>

#include <functional>
#include <map>

namespace nix {
namespace epoll {

class Flags {
 public:
  Flags(int flags);
  int Get();

 private:
  int flags_;
};

/// The tye of a callback registered to a file descriptor
typedef std::function<void (void)> Callback;


/// Stores a map of epoll event bitfields to their corresponding callback 
/// functions
class Subscription : public epoll_event {
 public:
  Subscription();

  /// Add callbacks to the registration
  void AddCallbacks(const std::map<int,Callback>& callbacks);
  
  /// Remove callbacks for the given bitset of events
  void RemoveCallbacks(int events);
  
  /// Called when a file descriptor event is occurs
  void Dispatch(int events);
  
  /// Return pointer to registration for epoll
  epoll_event* GetRegistration() { return &registration_; }
  
  /// Return the number of subscribed callbacks
  int NumCallbacks() { return callbacks_.size(); }
  
 private:
  /// event structure pointing pack to this object which is passed to
  /// the epoll instance
  epoll_event registration_;
  
  /// maps event bitfields to a callback function each
  std::map<int,Callback> callbacks_;
};

}  // namespace epoll


/// A dispatching wrapper for the linux epoll api
/**
 *  @see http://man7.org/linux/man-pages/man7/epoll.7.html
 *
 *  The Epoll class maintains a map from filedescriptors to Subscription
 *  objects. A Subscription is likewise a map from event id's (i.e. EPOLLIN,
 *  EPOLLOUT, EPOLLERR) to callbacks. Each call to `Wait()` is followed by
 *  iterating over all filedescriptors that are ready and dispatching callbacks
 *  for each subscribed event. 
 */
class Epoll {
 public:
  /// Create an epoll instance using epoll_create
  Epoll();

  /// Create an epoll instance using epoll_create1, while specifying flags
  Epoll(epoll::Flags flags);

  /// Closes the epoll file descriptor
  ~Epoll();

  /// Returns the underlying epoll file descriptor
  int GetFd() const;

  /// Add watches for the given file descriptor
  int Add(int fd, const std::map<int,epoll::Callback>& callbacks);

  /// Remove a watched events for the given file descriptor
  int Remove(int fd, int events);

  /// Block until one or more watched file descriptors are ready
  int Wait(int timeout) const;

  /// Block until one or more watched file descriptors are ready, and mask
  /// signals while blocked.
  int Pwait(int timeout, const sigset_t *sigmask) const;

 private:
  int epfd_;
  std::map<int,epoll::Subscription> subscriptions_;
  mutable std::vector<epoll_event> event_buffer_;
};

}  // namespace nix

#endif  // CPP_NIX_EPOLL_H_
