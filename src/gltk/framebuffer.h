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
 *  @date   Feb 3, 2013
 *  @author Josh Bialkowski (josh.bialkowski@gmail.com)
 *  @brief  
 */

#ifndef GLTK_FRAMEBUFFER_H_
#define GLTK_FRAMEBUFFER_H_

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <gltk/gluintref.h>
#include <gltk/refptr.h>

namespace gltk {

class FrameBuffer : public GLuintRef {
 private:
  /// calls glGenFramebuffers to instanciate a new framebuffer
  FrameBuffer();

 public:
  /// calls glDestroyFramebuffers to destroy the framebuffer
  ~FrameBuffer();

  /// binds the frame buffer so that future calls work with this
  /// buffer
  void Bind();

  /// create a frame buffer
  static RefPtr<FrameBuffer> Create();
};

}  // namespace gltk

#endif // GLTK_FRAMEBUFFER_H_
