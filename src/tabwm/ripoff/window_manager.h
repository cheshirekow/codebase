#ifndef TAB_WM_WINDOW_MANAGER_H_
#define TAB_WM_WINDOW_MANAGER_H_

#include <X11/Xlib.h>
#include <X11/Xutil.h>

#include <array>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#undef Success // defined by X.h, uh oh
#include <Eigen/Dense>
#include "util.h"


struct Node;

/// A node has two slots, and each one points to another node or to
/// an actual X window. It may also be empty, but only for the root node and
/// only while there are less than two windows on the screen.
struct Slot {
  enum {
    EMPTY, WINDOW, NODE
  } type;

  union {
    Window w;
    Node* n;
  } child;

  Slot() :
      type(EMPTY) {
  }
};

/// A node in the kd-tree used for layout.
struct Node {
  enum {
    HORIZONTAL, VERTICAL
  } split_type; ///< In which direction does this kd-tree node split.
  int split_loc; ///< The location of the split.
  std::array<Slot, 2> slots; ///< The two children of this node. They may be
                             ///  either other Nodes in the tree, or X windows
                             ///  themselves.

  Node() :
      split_type(HORIZONTAL), split_loc(0), slots() {
  }
};


/// Provides window management for an X display
class WindowManager {
public:
  /// Creates a WindowManager instance for the X display/screen specified by the
  /// argument string, or if unspecified, the DISPLAY environment variable. On
  /// failure, returns nullptr.
  static std::unique_ptr<WindowManager> Create(const std::string& display_str,
      int signal_fd);

  ~WindowManager();

  // Sets up signal handlers and enters the main loop
  void Run();

private:
  // File descriptor where we'll receive events if the process is signalled
  int signal_fd_;

  // Handle to the underlying Xlib Display struct.
  Display* display_;

  // Handle to root window.
  Window root_;

  // Attributes of each root window
  std::vector<XWindowAttributes> root_attr_;

  // Maps top-level windows to their frame windows.
  std::unordered_map<Window, Window> clients_;

  // The cursor position at the start of a window move/resize.
  Eigen::Vector2i drag_start_pos_;

  // The position of the affected window at the start of a window
  // move/resize.
  Eigen::Vector2i drag_start_frame_pos_;

  // The size of the affected window at the start of a window move/resize.
  Eigen::Vector2i drag_start_frame_size_;

  // Atom constants.
  Atom WM_PROTOCOLS;
  Atom WM_DELETE_WINDOW;

  // Invoked internally by Create().
  WindowManager(Display* display, int signal_fd);
  // Frames a top-level window.
  void Frame(Window w);
  // Unframes a client window.
  void Unframe(Window w);

  // Event handlers.
  void OnCreateNotify(const XCreateWindowEvent& e);
  void OnDestroyNotify(const XDestroyWindowEvent& e);
  void OnReparentNotify(const XReparentEvent& e);
  void OnMapNotify(const XMapEvent& e);
  void OnUnmapNotify(const XUnmapEvent& e);
  void OnConfigureNotify(const XConfigureEvent& e);
  void OnMapRequest(const XMapRequestEvent& e);
  void OnConfigureRequest(const XConfigureRequestEvent& e);
  void OnButtonPress(const XButtonEvent& e);
  void OnButtonRelease(const XButtonEvent& e);
  void OnMotionNotify(const XMotionEvent& e);
  void OnKeyPress(const XKeyEvent& e);
  void OnKeyRelease(const XKeyEvent& e);
};

#endif // TAB_WM_WINDOW_MANAGER_H_
