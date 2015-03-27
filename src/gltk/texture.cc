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

#include <gltk/texture.h>

namespace gltk {

Texture::Texture() {
  glGenTextures(1, &m_id);
}

Texture::~Texture() {
  glDeleteTextures(1, &m_id);
}

void Texture::Bind() {
  glBindTexture(GL_TEXTURE_2D, m_id);
}

RefPtr<Texture> Texture::Create() {
  return RefPtr<Texture>(new Texture());
}

}  // namespace gltk

