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

#ifndef GLTK_TEXTURE_H_
#define GLTK_TEXTURE_H_

#include <GL/glew.h>
#include <GL/glfw.h>
#include <gltk/gluintref.h>
#include <gltk/refptr.h>

namespace gltk {

class Texture : public GLuintRef {
 private:
  /// calls glGenTextures to instanciate a new texture
  Texture();

 public:
  /// calls glDestroyTextures to destroy the texture
  ~Texture();

  /// binds the texture so that future calls work with this texture
  void Bind();

  /// create a new texture
  static RefPtr<Texture> Create();
};

}  // namespace gltk

#endif // GLTK_TEXTURE_H_
