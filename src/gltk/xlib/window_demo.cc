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
 *  @date   Apr 14, 2015
 *  @author Josh Bialkowski (josh.bialkowski@gmail.com)
 *  @brief
 */

#include <unistd.h>
#include <gltk/xlib/window.h>

int main(int argc, char *argv[]) {
  std::shared_ptr<nix::Epoll> epoll_ptr(new nix::Epoll());
  std::shared_ptr<gltk::Pipeline> pipeline;
  std::unique_ptr<gltk::xlib::Window> window =
      gltk::xlib::Window::Create(epoll_ptr, pipeline);

  window->MakeCurrent();
  glClearColor(0, 0.5, 1, 1);
  glClear(GL_COLOR_BUFFER_BIT);
  window->SwapBuffers();

  sleep(1);

  window->MakeCurrent();
  glClearColor(1, 0.5, 0, 1);
  glClear(GL_COLOR_BUFFER_BIT);
  window->SwapBuffers();

  sleep(1);
  return 0;
}
