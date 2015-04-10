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
 *  @date Apr 9, 2015
 *  @author Josh Bialkowski
 */

#include <unistd.h>
#include <glog/logging.h>
#include <cpp_nix/epoll.h>


namespace nix {
namespace epoll {

Flags::Flags(int flags)
    : flags_(flags) {
}

int Flags::Get() {
  return flags_;
}

Subscription::Subscription() {
  registration_.data.ptr = this;
  registration_.events = 0;
}

void Subscription::AddCallbacks(const std::map<int,Callback>& callbacks) {
  for(auto& pair : callbacks) {
    // turn on the bit telling epoll we care about this event
    registration_.events |= pair.first;
    
    // store the callback for this event, perhaps over-writing an existing
    // event
    callbacks_[pair.first] = pair.second;
  }
}

void Subscription::RemoveCallbacks(int events) {
  registration_.events &= ~events;
  for(auto it = callbacks_.cbegin(); it != callbacks_.cend(); /* no iter */) {
    if(it->first & events) {
      callbacks_.erase(it++);
    } else {
      ++it;
    }
  }
}

void Subscription::Dispatch(int events) {
  for(auto& pair : callbacks_) {
    // if the event bitfield (pair.first) is active in the events bitvector
    // returned by epoll, then dispatch the callback
    if(events & pair.first) {
      pair.second();
    }
  }
}

}  // namespace epoll

Epoll::Epoll() {
  epfd_ = epoll_create(1);
}

Epoll::Epoll(epoll::Flags flags) {
  epfd_ = epoll_create1(flags.Get());
}

Epoll::~Epoll() {
  if (epfd_ > 0) {
    close(epfd_);
  }
}

int Epoll::GetFd() const {
  return epfd_;
}

int Epoll::Add(int fd, const std::map<int,epoll::Callback>& callbacks) {
  bool existing_subscription = subscriptions_.count(fd);
  subscriptions_[fd].AddCallbacks(callbacks);
  if(existing_subscription) {
    return epoll_ctl(epfd_, EPOLL_CTL_MOD, fd, 
                     subscriptions_[fd].GetRegistration());    
  } else {
    return epoll_ctl(epfd_, EPOLL_CTL_ADD, fd, 
                     subscriptions_[fd].GetRegistration());
  }
}

int Epoll::Remove(int fd, int events) {
  auto iter = subscriptions_.find(fd);
  if(iter == subscriptions_.end()) {
    LOG(WARNING) << "Attempt to remove events from  non-registred file "
                    "descriptor";
    return 1;
  } else {
    epoll::Subscription& subscription = iter->second;
    subscription.RemoveCallbacks(events);
    if(subscription.NumCallbacks() == 0) {
      subscriptions_.erase(iter);
      return epoll_ctl(epfd_, EPOLL_CTL_DEL, fd, NULL);
    }
  }
}

int Epoll::Wait(int timeout) const {
  event_buffer_.resize(subscriptions_.size());
  int retval = epoll_wait(epfd_, event_buffer_.data(), event_buffer_.size(),  
                          timeout);
  for(int i=0; i < retval; i++) {
    static_cast<epoll::Subscription*>(event_buffer_[i].data.ptr)
      ->Dispatch (event_buffer_[i].events);
  }
  return retval;
}

int Epoll::Pwait(int timeout, const sigset_t *sigmask) const {
  event_buffer_.resize(subscriptions_.size());
  int retval= epoll_pwait(epfd_, event_buffer_.data(), event_buffer_.size(), 
                          timeout, sigmask);
  for(int i=0; i < retval; i++) {
    static_cast<epoll::Subscription*>(event_buffer_[i].data.ptr)
      ->Dispatch (event_buffer_[i].events);
  }
  return retval;
}

}  // namespace nix

