/*
 *  Copyright (C) 2012 Josh Bialkowski (josh.bialkowski@gmail.com)
 *
 *  This file is part of gltk.
 *
 *  gltk is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  gltk is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with gltk.  If not, see <http://www.gnu.org/licenses/>.
 */
/**
 *  @file
 *  @date   Apr 10, 2015
 *  @author Josh Bialkowski (josh.bialkowski@gmail.com)
 *  @brief
 */
#ifndef GLTK_XLIB_XLIB_WINDOW_H_
#define GLTK_XLIB_XLIB_WINDOW_H_

#include <cstdint>
#include <memory>
#include <cpp_nix/epoll.h>

// forward declarations of X things we need to store
struct _XDisplay;
typedef _XDisplay Display;

namespace gltk {

/// A self-contained xlib window which provides xlib message translation to
/// `gltk` message types.
/**
 *  By default, each XlibWindow maintains it's own connection to the X server.
 *  If you would like to share a common connection to the server use the
 *  `Create()` method which takes a `Display` pointer.
 */
class XlibWindow {
 public:
  /// Calls XDestroyWindow
  ~XlibWindow();

  /// Create a new window with it's own connection to the display server
  static std::unique_ptr<XlibWindow> Create();

  /// Create a new window using the provided connection to the display
  /// server.
  static std::unique_ptr<XlibWindow> Create(
      const std::shared_ptr<Display>& display);

 private:
  /// construction only allowed through `Create()`
  XlibWindow(const std::shared_ptr<Display>& display, uint64_t window);

  std::shared_ptr<Display> display_;  ///< xserver connection
  uint64_t window_;                   ///< x11 window id
  nix::Epoll epoll_;                  ///< event multiplexer
};

}  // namespace gltk

#endif  // GLTK_XLIB_XLIB_WINDOW_H_
