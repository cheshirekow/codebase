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

#ifndef GLTK_BUFFER_H_
#define GLTK_BUFFER_H_

#include <GL/glew.h>
#include <GL/glfw.h>
#include <gltk/gluintref.h>
#include <gltk/refptr.h>

namespace gltk {

class Buffer : public GLuintRef {
 private:
  /// calls glGenBuffers to instanciate a new texture
  Buffer();

 public:
  /// calls glDestroyBuffers to destroy the texture
  ~Buffer();

  /// binds the texture so that future calls work with this texture
  void Bind(GLenum type);

  /// set data to this buffer
  void SetData(GLenum bufType, GLsizeiptr size, const void* data, GLenum usage);

  /// create a new texture
  static RefPtr<Buffer> Create();
};

}  // namespace gltk

#endif // GLTK_BUFFER_H_
