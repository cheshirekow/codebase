/**
 *  @file
 *  @date   Dec 23, 2012
 *  @author Josh Bialkowski (josh.bialkowski@gmail.com)
 *  @brief  
 */

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cstring>
#include <gltk/gltk.h>

const char* g_srcDir = "${CMAKE_CURRENT_SOURCE_DIR}";

int main(int argc, char** argv) {
  if (!glfwInit()) {
    std::cerr << "Failed to initialize GLFW" << std::endl;
    return -1;
  }

  glewExperimental = true;
  glfwWindowHint(GLFW_SAMPLES, 4);  // 4x antialiasing
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);  // We want OpenGL 3.3
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  //We don't want the old OpenGL

  // Open a window and create its OpenGL context
  GLFWwindow* window = glfwCreateWindow(1024, 768, "Test 01", NULL, NULL);
  if (!window) {
    fprintf( stderr, "Failed to open GLFW window\n");
    glfwTerminate();
    return -1;
  }

  // Initialize GLEW
  if (glewInit() != GLEW_OK) {
    fprintf(stderr, "Failed to initialize GLEW\n");
    return -1;
  }

  // Ensure we can capture the escape key being pressed below
  glfwSetInputMode(window, GLFW_STICKY_KEYS, GLFW_TRUE);

  // triangle data
  static const GLfloat vertex_buffer_data[] = { -1.0f, -1.0f, 0.0f, 1.0f, -1.0f,
      0.0f, 0.0f, 1.0f, 0.0f, };

  // create a triangle
  using namespace gltk;
  RefPtr<VertexArray> vertex_array = VertexArray::Create();
  vertex_array->Bind();

  RefPtr<Buffer> vertex_buffer = Buffer::Create();
  vertex_buffer->SetData(GL_ARRAY_BUFFER,
                         sizeof(vertex_buffer_data), vertex_buffer_data,
                         GL_STATIC_DRAW);

  if (glGetError() != GL_NO_ERROR) {
    std::cerr << "Failed to create vertex buffer" << std::endl;
  }

  // load shaders
  std::string info_log;
  std::string source;
  std::string source_file = std::string(g_srcDir)
                              + "/simple_vertex_shader.glslv";
  RefPtr<Shader> vertex_shader = Shader::CreateFromFile(GL_VERTEX_SHADER,
                                                        source_file.c_str());

  if (!vertex_shader || glGetError() != GL_NO_ERROR) {
    std::cerr << "Failed to load shader file: " << source_file << std::endl;
    return 1;
  }

  vertex_shader->compile();
  if (!vertex_shader->GetCompileStatus()) {
    vertex_shader->GetSource(source);
    vertex_shader->GetInfoLog(info_log);
    std::cerr << "Failed to compile vertex shader: " << info_log << "\n-----\n"
              << source << std::endl;
  }

  source_file = std::string(g_srcDir) + "/simple_fragment_shader.glslf";
  RefPtr<Shader> fragment_shader = Shader::CreateFromFile(GL_FRAGMENT_SHADER,
                                                          source_file.c_str());
  if (!fragment_shader  || glGetError() != GL_NO_ERROR) {
    std::cerr << "Failed to load shader file: " << source_file << std::endl;
    return 1;
  }

  fragment_shader->compile();
  if (!fragment_shader->GetCompileStatus()) {
    fragment_shader->GetSource(source);
    fragment_shader->GetInfoLog(info_log);
    std::cerr << "Failed to compile fragment shader: " << info_log << "\n-----\n"
              << source << std::endl;
  }

  RefPtr<Program> program = Program::Create();
  program->AttachShader(vertex_shader);
  program->AttachShader(fragment_shader);
  program->Link();

  if (!program->GetLinkStatus()) {
    program->GetInfoLog(info_log);
    std::cerr << "Failed to link shader program: " << info_log << std::endl;
  }

  vertex_shader.Unlink();
  fragment_shader.Unlink();

  do {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    program->Use();

    // Draw nothing, see you in tutorial 2 !
    glEnableVertexAttribArray(0);
    vertex_buffer->Bind(GL_ARRAY_BUFFER);
    glVertexAttribPointer(0,  // attribute 0. No particular reason for 0, but must match the layout in the shader.
        3,                  // size
        GL_FLOAT,           // type
        GL_FALSE,           // normalized?
        0,                  // stride
        (void*) 0            // array buffer offset
        );

    // Draw the triangle !
    glDrawArrays(GL_TRIANGLES, 0, 3);  // Starting from vertex 0; 3 vertices total -> 1 triangle

    glDisableVertexAttribArray(0);

    // Swap buffers
    glfwSwapBuffers(window);

  }  // Check if the ESC key was pressed or the window was closed
  while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS
      && !glfwWindowShouldClose(window));
}



