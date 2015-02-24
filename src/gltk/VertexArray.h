/*
 *  Copyright (C) 2012 Josh Bialkowski (jbialk@mit.edu)
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
 *  @file   /home/josh/Codes/cpp/gltk/src/gltk/VertexArray.h
 *
 *  @date   Feb 3, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef GLTK_VERTEXARRAY_H_
#define GLTK_VERTEXARRAY_H_


#include <GL/glew.h>
#include <GL/glfw.h>
#include <gltk/GLuintRef.h>
#include <gltk/RefPtr.h>

namespace gltk {

class VertexArray:
    public GLuintRef
{
    private:
        /// calls glGenVertexArrays to instanciate a new texture
        VertexArray();

    public:
        /// calls glDestroyVertexArrays to destroy the texture
        ~VertexArray();

        /// binds the texture so that future calls work with this texture
        void bind( );

        /// create a new texture
        static RefPtr<VertexArray> create();
};


} // namespace gltk



#endif // VERTEXARRAY_H_
