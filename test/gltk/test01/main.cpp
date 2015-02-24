/**
 *  @file   src/tut01/main.cpp
 *
 *  @date   Dec 23, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <GL/glew.h>
#include <GL/glfw.h>
#include <cstring>
#include <gltk.h>

const char* g_srcDir = "${CMAKE_CURRENT_SOURCE_DIR}";

int main(int argc, char** argv)
{
    if( !glfwInit() )
    {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    glewExperimental = true;
    glfwOpenWindowHint(GLFW_FSAA_SAMPLES, 4); // 4x antialiasing
    glfwOpenWindowHint(GLFW_OPENGL_VERSION_MAJOR, 3); // We want OpenGL 3.3
    glfwOpenWindowHint(GLFW_OPENGL_VERSION_MINOR, 3);
    glfwOpenWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); //We don't want the old OpenGL

    // Open a window and create its OpenGL context
    if( !glfwOpenWindow( 1024, 768, 0,0,0,0, 32,0, GLFW_WINDOW ) )
    {
        fprintf( stderr, "Failed to open GLFW window\n" );
        glfwTerminate();
        return -1;
    }

    // Initialize GLEW
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Failed to initialize GLEW\n");
        return -1;
    }

    glfwSetWindowTitle( "Test 01" );

    // Ensure we can capture the escape key being pressed below
    glfwEnable( GLFW_STICKY_KEYS );

    // triangle data
    static const GLfloat vertex_buffer_data[] =
    {
        -1.0f, -1.0f, 0.0f,
         1.0f, -1.0f, 0.0f,
         0.0f,  1.0f, 0.0f,
    };

    // create a triangle
    using namespace gltk;
    RefPtr<VertexArray> vertexArray = VertexArray::create();
    vertexArray->bind();

    RefPtr<Buffer>      vertexBuffer = Buffer::create();
    vertexBuffer->setData(
            GL_ARRAY_BUFFER, sizeof(vertex_buffer_data),
            vertex_buffer_data, GL_STATIC_DRAW);

    if( glGetError != GL_NO_ERROR )
    {
        std::cerr << "Failed to create vertex buffer" << std::endl;
    }

    // load shaders
    std::string infoLog;
    std::string source;
    RefPtr<Shader> vertexShader =
            Shader::create_from_file( GL_VERTEX_SHADER,
        (std::string(g_srcDir) + "/SimpleVertexShader.vertexshader").c_str());

    if(!vertexShader || glGetError != GL_NO_ERROR)
    {
        std::cerr << "Failed to load shader file: "
                  << std::string(g_srcDir) + "/SimpleVertexShader.vertexshader"
                  << std::endl;
        return 1;
    }

    vertexShader->compile();
    if( !vertexShader->getCompileStatus() )
    {
        vertexShader->getSource(source);
        vertexShader->getInfoLog(infoLog);
        std::cerr << "Failed to compile vertex shader: "
                    << infoLog
                    << "\n-----\n"
                    << source << std::endl;
    }

    RefPtr<Shader> fragmentShader =
            Shader::create_from_file( GL_FRAGMENT_SHADER,
        (std::string(g_srcDir) + "/SimpleFragmentShader.fragmentshader").c_str());

    if(!fragmentShader)
    {
        std::cerr << "Failed to load shader file: "
              << std::string(g_srcDir) + "/SimpleFragmentShader.fragmentshader"
              << std::endl;
        return 1;
    }

    fragmentShader->compile();
    if( !fragmentShader->getCompileStatus() )
    {
        fragmentShader->getSource(source);
        fragmentShader->getInfoLog(infoLog);
        std::cerr << "Failed to compile fragment shader: "
                    << infoLog
                    << "\n-----\n"
                    << source << std::endl;
    }

    RefPtr<Program> program = Program::create();
    program->attachShader(vertexShader);
    program->attachShader(fragmentShader);
    program->link();

    if( !program->getLinkStatus() )
    {
        program->getInfoLog(infoLog);
        std::cerr << "Failed to link shader program: "
                    << infoLog << std::endl;
    }

    vertexShader.unlink();
    fragmentShader.unlink();


    do{
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        program->use();

        // Draw nothing, see you in tutorial 2 !
        glEnableVertexAttribArray(0);
        vertexBuffer->bind(GL_ARRAY_BUFFER);
        glVertexAttribPointer(
           0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
           3,                  // size
           GL_FLOAT,           // type
           GL_FALSE,           // normalized?
           0,                  // stride
           (void*)0            // array buffer offset
        );

        // Draw the triangle !
        glDrawArrays(GL_TRIANGLES, 0, 3); // Starting from vertex 0; 3 vertices total -> 1 triangle

        glDisableVertexAttribArray(0);

        // Swap buffers
        glfwSwapBuffers();

    } // Check if the ESC key was pressed or the window was closed
    while( glfwGetKey( GLFW_KEY_ESC ) != GLFW_PRESS &&
            glfwGetWindowParam( GLFW_OPENED ) );
}



