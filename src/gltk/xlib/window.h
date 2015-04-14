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
#ifndef GLTK_XLIB_WINDOW_H_
#define GLTK_XLIB_WINDOW_H_

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <GL/gl.h>
#include <GL/glx.h>

#include <cstdint>
#include <memory>
#include <cpp_nix/epoll.h>

namespace gltk {
namespace xlib {

/// A self-contained xlib window which provides xlib message translation to
/// `gltk` message types.
/**should
 *  By default, each XlibWindow maintains it's own connection to the X server.
 *  If you would like to share a common connection to the server use the
 *  `Create()` method which takes a `Display` pointer.
 */
class Window {
 public:
  /// Calls XDestroyWindow
  ~Window();

  /// Create a new window with it's own connection to the display server
  /**
   * @param epoll   the epoll instance to attach to
   *
   * If the window is created successfully then it will attach the socket
   * file descriptor for it's X11 server connection to the specified epoll
   * instance so that it may process X11 events when they occur.
   */
  static std::unique_ptr<Window> Create(
      const std::shared_ptr<nix::Epoll>& epoll);

  /// Reads all available messages in the xlib inbound queue and dispatches
  /// them to the appropriate handler
  void DispatchXEvents();
  void DoDemo();

 private:
  /// construction only allowed through `Create()`
  Window(Display* display, GLXContext context, Colormap color_map,
         ::Window window);

  Display* display_;     ///< xserver connection
  GLXContext context_;  ///< glx context
  Colormap color_map_;  ///< x11 color map
  ::Window window_;        ///< x11 window id
};

}  // namespace xlib
}  // namespace gltk

#endif  // GLTK_XLIB_WINDOW_H_
