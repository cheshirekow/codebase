#include <unistd.h>
#include <csignal>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include "window_manager.h"

int g_signal_fd = 0;

// If the process receives a signal, just dump it into the write end of a pipe
// so that the main loop can process it asynchronously.
void OnProcessSignal(int signum) {
  char signum_as_char = static_cast<char>(signum);
  if (write(g_signal_fd, &signum_as_char, 1) < 1) {
    LOG(ERROR)<< "Failed to write signal to pipe";
  }
}

DEFINE_string(display, "",
    "The X display server to use, overrides DISPLAY environment variable.");

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  // create a pipe where we can dump signals that we are catching. This will
  // allow our main event loop to select on both the X connection file
  // descriptor, and the file descriptor for signals.
  int pipe_ends[2];
  CHECK_EQ(0,pipe(pipe_ends));
  g_signal_fd = pipe_ends[1];

  // Set up our signal handler
  signal(SIGINT, OnProcessSignal);

  // Construct the window manager
  std::unique_ptr<WindowManager> window_manager(
      WindowManager::Create(FLAGS_display, pipe_ends[0]));
  if (!window_manager) {
    LOG(ERROR)<< "Failed to initialize window manager.";
    return 1;
  }

  // Run the main loop until termination.
  window_manager->Run();
  return 0;
}
