#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <iostream>
#include <vector>

std::vector<char*> MakeArgv(std::vector<std::string>* source) {
  std::vector<char*> out;
  out.reserve(source->size() + 1);
  for (std::string& str : *source) {
    out.push_back(&str[0]);
  }
  out.push_back(nullptr);
  return out;
}

int main(int argc, char* argv[], char* envp[]) {
  pid_t xephyr_child = fork();
  if (xephyr_child == 0) {
    std::vector<std::string> argv_str = { "Xephyr"
        "-ac", "-br", "-noreset", "+extension", "GLX", "-screen", "800x600",
        ":1" };
    std::vector<char*> argv = MakeArgv(&argv_str);
    int result = execv("/usr/bin/Xephyr", &argv[0]);
    if (result < 0) {
      std::cerr << "Failed to spawn xephyr\n";
      return 1;
    }
  }
  usleep(1e6);

  pid_t glx_gears_child = fork();
  if (glx_gears_child == 0) {
    std::vector<std::string> argv_str = { "glxgears", "-display", ":1" };
    std::vector<char*> argv = MakeArgv(&argv_str);
    int result = execv("/usr/bin/glxgears", &argv[0]);
    if (result < 0) {
      std::cerr << "Failed to spawn glxgears\n";
      return 1;
    }
  }

  pid_t xeyes_child = fork();
  if (xeyes_child == 0) {
    std::vector<std::string> argv_str = { "xeyes", "-display", ":1" };
    std::vector<char*> argv = MakeArgv(&argv_str);
    int result = execv("/usr/bin/xeyes", &argv[0]);
    if (result < 0) {
      std::cerr < "Failed to spawn xeyes\n";
    }
  }

  pid_t tab_wm_child = fork();
  if (tab_wm_child == 0) {
    std::vector<std::string> argv_str = { "tab_wm", "--display=:1", };
    std::vector<char*> argv = MakeArgv(&argv_str);
    std::string full_path = "@CMAKE_CURRENT_BINARY_DIR@/../src/tab_wm";
    int result = execv(full_path.c_str(), &argv[0]);
    if (result < 0) {
      std::cerr << "Failed to spawn tab_wm from " << full_path << "\n";
      return 1;
    }
  }

  int status = 0;
  int options = 0;
  waitpid(xephyr_child, &status, options);
  waitpid(glx_gears_child, &status, options);
  waitpid(xeyes_child, &status, options);
  waitpid(tab_wm_child, &status, options);
}
