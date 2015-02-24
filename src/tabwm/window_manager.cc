#include <fcntl.h>
#include <unistd.h>
#include <sys/epoll.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xatom.h>

#include <algorithm>
#include <array>
#include <bitset>
#include <csignal>
#include <cstring>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>

#undef Success // defined by X.h, uh oh
#include <boost/circular_buffer.hpp>
#include <Eigen/Dense>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include "util.h"

bool g_wm_detected = false; ///< May be set to true when we try to make
                            ///  ourselves the window manager, indicataing that
                            ///  another window manager is already running

int g_signal_write_fd = 0;  ///< The file descriptor of the write-end of a pipe
                            ///  where our signal handler pushes a one-byte
                            ///  cast of the signal number when our process is
                            ///  signaled.

int g_signal_read_fd = 0;   ///< The file descriptor of the read-end of a pipe
                            ///  where the signal handler pumps signals

int g_epoll_fd = 0; ///< The file descriptor of our epoll instance

int g_listen_fd = 0; ///< unix domain socket to listen on

bool g_should_quit = false; ///< Set to true when the process should exit

// Command line flags:
DEFINE_string(display, "",
    "The X display server to use, overrides DISPLAY environment variable.");
DEFINE_int32(screen, 0,
    "Which X screen to manage, if there are multiple X screens in play.");
DEFINE_string(socket, "/tmp/tabwm_socket",
    "Path to the unix domain socket to open and listen on.");

enum EpollType {
  EPOLL_SIGNAL_FD,
  EPOLL_X_FD,
  EPOLL_LISTEN_SOCKET_FD,
  EPOLL_CLIENT_SOCKET_FD,
};

struct EpollData {
  EpollType type;
  int fd;
};

struct Node;

/// A kd-node within a tree can be referenced by the "path" to that node from
/// the root. At each node, the tree branches exactly twice so which branch we
/// take can be encoded in a single bit.
typedef std::vector<bool> TreePath;

/// A node in the kd-tree used for layout.
struct Node {
  typedef std::unique_ptr<Node> Ptr;

  enum {
    HORIZONTAL, VERTICAL
  } split_type;  ///< In which direction does this kd-tree node split.
  int split_loc; ///< The location of the split in pixels.
  Window window;  ///< the window at this node, is set to 0 if this is not
                  ///  a leaf
  std::array<Ptr, 2> children; ///< The two children of this node.

  Node() :
      split_type(HORIZONTAL), split_loc(0), window(0) {
  }

  Node(Window w) :
      split_type(HORIZONTAL), split_loc(0), window(w) {
  }

  /// Assume the data of another node
  void Assume(Node* other) {
    // The parent node assumes the data of the sibling node, essentially moving
    // the sibling node up one level
    split_loc = other->split_loc;
    split_type = other->split_type;
    window = other->window;
    children[0].reset(other->children[0].release());
    children[1].reset(other->children[1].release());
  }
};

/// Return the node at the specified path.
Node* Find(Node* root, TreePath path) {
  Node* node = root;
  for (bool which_direction : path) {
    if (node) {
      node = node->children[which_direction].get();
    } else {
      return nullptr;
    }
  }
  return node;
}

void RemoveNode(Node* root_node, TreePath tree_path) {
  assert(tree_path.size() > 2);
  bool child_path = tree_path.back();
  tree_path.pop_back();
  bool parent_path = tree_path.back();
  tree_path.pop_back();

  Node* grand_parent = Find(root_node, tree_path);
  CHECK_NOTNULL(grand_parent);
  Node* parent = grand_parent->children[parent_path];
  CHECK_NOTNULL(parent);
  std::unique_ptr<Node> sibling(parent->children[!child_path].release());
  CHECK_NOTNULL(sibling.get());
  grand_parent->children[parent_path].reset(sibling.release());
}

struct Workspace {
  std::unique_ptr<Node> tree_root;
  std::list<Window> floating;
};

void RemoveCell(Workspace* w, TreePath tree_path) {
  switch (tree_path.size()) {
  case 0:
    w->tree_root.reset();
    break;
  case 1:
    w->tree_root.reset(w->tree_root->children[!tree_path[0]].get());
  }
}

/// Meta info that we store for each top level window that we know about
struct WindowInfo {
  /// Which layer the window is on
  enum {
    FLOATING,
    TREE
  } layer;

  /// Iterator pointing to the workspace (i.e. kd-tree root) on which this
  /// window lies.
  std::list<Workspace>::iterator  workspace;
  std::list<Window>::iterator float_path;

  /// Path to the window within the tree.
  TreePath tree_path;

  XWMHints x_wm_hints;
  XClassHint x_wm_class_hint;
  std::map<Atom, std::string> x_wm_text;
  Window   transient_for ;


  /// Called when a new window tries to map itself, Xserver information
  /// about the window to try and determine which layer of which workspace
  /// it goes on
  void SetupWindow(Window w) {
    // select input before grabbing properties to prevent race condition. We
    // should get all property changes after this
    XSelectInput(g_display, w, PropertyChangeMask);

    transient_for = 0;
    LOG_IF(WARNING, XGetTransientForHint(g_display, w, &transient_for)==0)
      << "Failed to get transient-for hint for window "
      << w;

    /// Grab text properties that we are interested, and store them in a map
    /// according to their atom
    for (Atom atom : { XA_WM_ICON_NAME, XA_WM_NAME}) {
      XTextProperty text_prop;
      if (XGetTextProperty(g_display, w, &text_prop, WM_NAME) > 0) {
        if (text_prop.value) {
          x_wm_text[atom] = text_prop.value;
          XFree(text_prop.value);
        }
      } else {
        LOG(WARNING)<< "Failed to retrieve WM_NAME for window " << w;
      }
    }
    x_wm_class_hint = {nullptr, nullptr};
    LOG_IF(WARNING, XGetClassHint(g_display,w,&x_wm_class_hint)==0)
      << "Failed to retrieve WM_CLASS";

    /// If this is a transient window, then find the associated top-level
    /// window, use it's workspace, and put this new window on the floating
    /// layer
    if(transient_for != 0) {

    }

  }
};

/// Opaque handle to our X server connections
Display* g_display;

/// The root window of our screen
Window g_root_win;

/// Attributes of the root window
XWindowAttributes g_root_attr;

/// List of active workspaces
std::list<Workspace> g_workspaces;

/// The workspace that is currently active
std::list<Workspace>::iterator g_current_workspace;

/// Maps each window we know about to a structure of their meta data
std::map<Window, WindowInfo> g_metadata;

/// Maps top-level windows to their workspace
std::map<Window,WindowInfo> window_info_;

// Atom constants.
Atom WM_PROTOCOLS;
Atom WM_DELETE_WINDOW;

/// When a window is created, put it on a new workspace and create a info struct
/// for that window so we can track it.
void OnCreateNotify(const XCreateWindowEvent& e) {
  // ignore decorator windows, we will handle them when the child is reparented
  std::list<Workspace>::iterator workspace = g_workspaces.emplace(
      g_workspaces.back());

  Window transient_for_window = 0;
  if(XGetTransientForHint(g_display, e.window, &transient_for_window)){
    std::list<Window>::iterator floating_iter = workspace->floating.push_back(
        e.window);
    window_info_.emplace(e.window, workspace, floating_iter);
  } else {
    workspace->tree_root.reset(new Node(e.window));
    window_info_.emplace(e.window, workspace, TreePath());
  }
}



/// When a window is destroyed, find which workspace it is on, and remove it
/// from the tree
void OnDestroyNotify(const XDestroyWindowEvent& e) {
  auto iter = window_info_.find(e.window);
  if (iter == window_info_.end()) {
    LOG(WARNING)<< "Untracked window [" << e.window << "] was destroyed.";
  } else {
    WindowInfo& win_info = iter->second;
    std::list<Workspace>::iterator workspace = win_info.workspace;
    if(win_info.layer == WindowInfo::FLOATING) {
      workspace->floating.erase(win_info.float_path);
    } else {
      RemoveNode(workspace->tree_root.get(),win_info.tree_path);
    }
    if(workspace->floating.size() == 0 && !workspace->tree_root) {
      g_workspaces.erase(workspace);
    }
    window_info_.erase(iter);
  }
}

/// When a window is reparented by the decorator, move the decorated window
/// into the vacated slot.
void OnReparentNotify(const XReparentEvent& e) {
  Window decorator  = e.parent;
  Window app_window = e.window;
  auto iter = window_info_.find(e.window);
  if(iter == window_info_.end()) {
    LOG(WARNING) << "Window [" << e.window << "] was reparented to ["
        << e.parent << "] but I don't have an info structure for it";
    return;
  } else {
    window_info_[e.parent] = iter->second;
    window_info_.erase(iter);
  }
}


void OnMapNotify(const XMapEvent& e) {}

/// When a window wants to be hidden (i.e. iconified) we remove the frame from
/// from the window.
void OnUnmapNotify(const XUnmapEvent& e) {
  // If the window is a client window we manage, unframe it upon UnmapNotify. We
  // need the check because other than a client window, we can receive an
  // UnmapNotify for
  //     - A frame we just destroyed ourselves.
  //     - A pre-existing and mapped top-level window we reparented.
  if (!frame_of_client_.count(e.window)) {
    LOG(INFO) << "Ignore UnmapNotify for non-client window " << e.window;
    return;
  }

  // Since we selected substructure notify in order to receive this event,
  // according to the manpage for XUnmapEvent, this means that e.event will
  // be set to the parent of the unmapped window and e.window will be set to
  // the unmapped window. Therefore, if e.event is the handle to one of the
  // root windows of a screen, that means that this window was
  // not framed by us and so we should do nothing. Note that I believe this
  // may be possible if the window had the override_redirect flag set when
  if (e.event == g_root_win) {
    LOG(INFO) << "Ignore UnmapNotify for reparented pre-existing window "
    << e.window;
    return;
  }

  Unframe(e.window);
}

void OnConfigureNotify(const XConfigureEvent& e) {}

/// When a window wants to be shown we add a frame to it and then allow it to
/// be shown.
void OnMapRequest(const XMapRequestEvent& e) {
  Frame(e.window);
  XMapWindow(g_display, e.window);
}

/// If a window wishes to changes size or location, then we make the bounding
/// frame match the new size/location, if it exists.
void OnConfigureRequest(const XConfigureRequestEvent& e) {
  XWindowChanges changes;
  changes.x = e.x;
  changes.y = e.y;
  changes.width = e.width;
  changes.height = e.height;
  changes.border_width = e.border_width;
  changes.sibling = e.above;
  changes.stack_mode = e.detail;
  if (frame_of_client_.count(e.window)) {
    const Window frame = frame_of_client_[e.window];
    XConfigureWindow(g_display, frame, e.value_mask, &changes);
    LOG(INFO)<< FormatStr("Resize [%ul] to (%d,%d)", frame, e.width, e.height);
  }
  XConfigureWindow(g_display, e.window, e.value_mask, &changes);
  LOG(INFO)<< FormatStr("Resize %ul to (%d,%d)", e.window, e.width, e.height);
}

/// When we receive a button press event we save the initial geometry of the
/// window with focus because it means the window is either going to be
/// resized or moved.
void OnButtonPress(const XButtonEvent& e) {
  CHECK(frame_of_client_.count(e.window));
  const Window frame = frame_of_client_[e.window];

  // Save initial cursor position.
  drag_start_pos_ = Eigen::Vector2i(e.x_root, e.y_root);

  // Save initial window info.
  Window returned_root;
  int x, y;
  unsigned width, height, border_width, depth;

  CHECK(
      XGetGeometry(g_display, frame, &returned_root, &x, &y, &width, &height,
                   &border_width, &depth));
  drag_start_frame_pos_ = Eigen::Vector2i(x, y);
  drag_start_frame_size_ = Eigen::Vector2i(width, height);

  // Raise clicked window to top.
  XRaiseWindow(g_display, frame);
}

void OnButtonRelease(const XButtonEvent& e) {}

void OnMotionNotify(const XMotionEvent& e) {
  CHECK(frame_of_client_.count(e.window));
  const Window frame = frame_of_client_[e.window];
  const Eigen::Vector2i drag_pos(e.x_root, e.y_root);
  const Eigen::Vector2i delta = drag_pos - drag_start_pos_;

  if (e.state & Button1Mask) {
    // alt + left button: Move window.
    const Eigen::Vector2i dest_frame_pos = drag_start_frame_pos_ + delta;
    XMoveWindow(g_display, frame, dest_frame_pos[0], dest_frame_pos[1]);
  } else if (e.state & Button3Mask) {
    // alt + right button: Resize window.
    // Window dimensions cannot be negative.
    const Eigen::Vector2i size_delta(
        std::max(delta[0], -drag_start_frame_size_[0]),
        std::max(delta[1], -drag_start_frame_size_[1]));
    const Eigen::Vector2i dest_frame_size = drag_start_frame_size_ + size_delta;

    // Resize frame.
    XResizeWindow(g_display, frame, dest_frame_size[0], dest_frame_size[1]);

    // Resize client window.
    XResizeWindow(g_display, e.window, dest_frame_size[0], dest_frame_size[1]);
  }
}

void OnKeyPress(const XKeyEvent& e) {
  if ((e.state & Mod1Mask)
      && (e.keycode == XKeysymToKeycode(g_display, XK_F4))) {
    // alt + f4: Close window.
    //
    // There are two ways to tell an X window to close. The first is to send it
    // a message of type WM_PROTOCOLS and value WM_DELETE_WINDOW. If the client
    // has not explicitly marked itself as supporting this more civilized
    // behavior (using XSetWMProtocols()), we kill it with XKillClient().
    Atom* supported_protocols;
    int num_supported_protocols;
    if (XGetWMProtocols(g_display, e.window, &supported_protocols,
                        &num_supported_protocols)
        && (std::find(supported_protocols,
                      supported_protocols + num_supported_protocols,
                      WM_DELETE_WINDOW)
            != supported_protocols + num_supported_protocols)) {
      LOG(INFO)<< "Gracefully deleting window " << e.window;
      // 1. Construct message.
      XEvent msg;
      memset(&msg, 0, sizeof(msg));
      msg.xclient.type = ClientMessage;
      msg.xclient.message_type = WM_PROTOCOLS;
      msg.xclient.window = e.window;
      msg.xclient.format = 32;
      msg.xclient.data.l[0] = WM_DELETE_WINDOW;
      // 2. Send message to window to be closed.
      CHECK(XSendEvent(g_display, e.window, false, 0, &msg));
    } else {
      LOG(INFO) << "Killing window " << e.window;
      XKillClient(g_display, e.window);
    }
  } else if ((e.state & Mod1Mask) &&
      (e.keycode == XKeysymToKeycode(g_display, XK_Tab))) {
    // alt + tab: Switch window.
    // 1. Find next window.
    auto i = frame_of_client_.find(e.window);
    CHECK(i != frame_of_client_.end());
    ++i;
    if (i == frame_of_client_.end()) {
      i = frame_of_client_.begin();
    }
    // 2. Raise and set focus.
    XRaiseWindow(g_display, i->second);
    XSetInputFocus(g_display, i->first, RevertToPointerRoot, CurrentTime);
  }
}

void OnKeyRelease(const XKeyEvent& e) {}

/// Xlib error handler.
int OnXError(Display* display, XErrorEvent* e) {
  const int MAX_ERROR_TEXT_LENGTH = 1024;
  char error_text[MAX_ERROR_TEXT_LENGTH];
  XGetErrorText(display, e->error_code, error_text, sizeof(error_text));
  LOG(ERROR) << "Received X error:\n"
             << "    Request: " << int(e->request_code)
             << " - " << XRequestCodeToString(e->request_code) << "\n"
             << "    Error code: " << int(e->error_code)
             << " - " << error_text << "\n"
             << "    Resource ID: " << e->resourceid;
  return 0;
}

/// Xlib error handler installed when we attempt to register ourselves as the
/// window manager. If X refuses our request it will callback on this
/// error handler, which will mark that another window manager has been
/// detected
int OnWMDetected(Display* display, XErrorEvent* e) {
  // In the case of an already running window manager, the error code from
  // XSelectInput is BadAccess. We don't expect this handler to receive any
  // other errors.
  CHECK_EQ(static_cast<int>(e->error_code), BadAccess);
  // Set flag.
  g_wm_detected = true;
  // The return value is ignored.
  return 0;
}


// If the process receives a signal, just dump it into the write end of a pipe
// so that the main loop can process it asynchronously.
void OnProcessSignal(int signum) {
  char signum_as_char = static_cast<char>(signum);
  if (write(g_signal_write_fd, &signum_as_char, 1) < 1) {
    LOG(ERROR)<< "Failed to write signal to pipe";
  }
}

void DispatchEvent(XEvent* event) {
  switch (event->type) {
    case CreateNotify:
      OnCreateNotify(event->xcreatewindow);
      break;
    case DestroyNotify:
      OnDestroyNotify(event->xdestroywindow);
      break;
    case ReparentNotify:
      OnReparentNotify(event->xreparent);
      break;
    case MapNotify:
      OnMapNotify(event->xmap);
      break;
    case UnmapNotify:
      OnUnmapNotify(event->xunmap);
      break;
    case ConfigureNotify:
      OnConfigureNotify(event->xconfigure);
      break;
    case MapRequest:
      OnMapRequest(event->xmaprequest);
      break;
    case ConfigureRequest:
      OnConfigureRequest(event->xconfigurerequest);
      break;
    case ButtonPress:
      OnButtonPress(event->xbutton);
      break;
    case ButtonRelease:
      OnButtonRelease(event->xbutton);
      break;
    case MotionNotify:
      // Skip any already pending motion events.
      while (XCheckTypedWindowEvent(g_display, event->xmotion.window,
      MotionNotify,
                                    event)) {
      }
      OnMotionNotify(event->xmotion);
      break;
    case KeyPress:
      OnKeyPress(event->xkey);
      break;
    case KeyRelease:
      OnKeyRelease(event->xkey);
      break;
    default:
      LOG(WARNING)<< "Ignored event";
    }
  }

void PumpXQueue() {
  while (XPending(g_display)) {
    XEvent e;
    XNextEvent(g_display, &e);
    LOG(INFO)<< "Received event: " << XEventToString(e);
    DispatchEvent(&e);
  }
}

void DispatchEpoll(epoll_event& event) {
  EpollData* data = static_cast<EpollData*>(event.data.ptr);
  switch (data->type) {
  case EPOLL_X_FD: {
    PumpXQueue();
    break;
  }

  case EPOLL_SIGNAL_FD: {
    g_should_quit = true;
    break;
  }

  case EPOLL_LISTEN_SOCKET_FD : {
    int client_fd = accept4(g_listen_fd, nullptr, 0, SOCK_NONBLOCK);
    if(client_fd == -1) {
      LOG(WARNING) << "Failed to accept client connection: " << strerror(errno);
    } else {
      epoll_event epoll_spec;
      epoll_spec.events = EPOLLIN;
      epoll_spec.data.ptr = new EpollData { EPOLL_CLIENT_SOCKET_FD, client_fd };
        LOG_IF(WARNING, -1 ==
            epoll_ctl(g_epoll_fd, EPOLL_CTL_ADD, g_signal_read_fd, &epoll_spec))
          << "epoll_ctl: " << strerror(errno);
    }
    break;
  }

  default:
    break;
  }
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  // create a pipe where we can dump signals that we are catching. This will
  // allow our main event loop to select on both the X connection file
  // descriptor, and the file descriptor for signals.
  int pipe_ends[2];

  CHECK_EQ(0,pipe(pipe_ends)) << "Failed to create a pipe: " << strerror(errno);
  g_signal_read_fd = pipe_ends[0];
  g_signal_write_fd = pipe_ends[1];

  // Set up our signal handler
  signal(SIGINT, OnProcessSignal);
  signal(SIGHUP, SIG_IGN);

  // Set up our epoll instance
  g_epoll_fd = epoll_create1(0);
  CHECK_NE(-1, g_epoll_fd) << "Failed to create an epoll instance: "
      << strerror(errno);

  // Set up our IPC socket
  g_listen_fd = socket(AF_UNIX, SOCK_STREAM, 0);
  CHECK_NE(-1, g_listen_fd) << "Failed to open a UNIX domain socket : "
      << strerror(errno);
  int flags = fcntl(g_listen_fd, F_GETFL, 0);
  CHECK_NE(-1, flags) << "Failed to get flag  on listener fd: "
      << strerror(errno);
  flags |= O_NONBLOCK;
  CHECK_NE(-1, fcntl(g_listen_fd, F_SETFL, flags))
    << "Failed to set non-blocking flag on listening fd: " << strerror(errno);

  LOG_IF(WARNING, -1 == unlink(FLAGS_socket.c_str()))
   << "Failed to unlink the socket file "
   << FLAGS_socket << " : "
   << strerror(errno);

  sockaddr_un local_addr;
  local_addr.sun_family = AF_UNIX;
  CHECK_LT(FLAGS_socket.size(), sizeof(local_addr.sun_path))
    << "Unix socket path is limited to " << sizeof(local_addr.sun_path)
    << " characters";
  strcpy(local_addr.sun_path, FLAGS_socket.c_str());
  CHECK_NE(-1, bind(g_listen_fd, (sockaddr*)&local_addr, sizeof(sockaddr_un)))
    << "bind : " << strerror(errno);
  CHECK_NE(-1, listen(g_listen_fd, 5))<< "listen : " << strerror(errno);

  // Open connection to  X display.server
  const char* display_c_str =
      FLAGS_display.empty() ? nullptr : FLAGS_display.c_str();
  LOG(INFO)<< "Opening X display " << FLAGS_display;
  g_display = XOpenDisplay(display_c_str);
  if (g_display == nullptr) {
    LOG(ERROR)<< "Failed to open X display " << XDisplayName(display_c_str);
    return 1;
  }

  // initialize program state
  WM_PROTOCOLS = XInternAtom(g_display, "WM_PROTOCOLS", false);
  WM_DELETE_WINDOW = XInternAtom(g_display, "WM_DELETE_WINDOW", false);

  // Get the window attributes for the root window on each screen
  int num_screens = ScreenCount(g_display);
  if (FLAGS_screen < 0 || FLAGS_screen >= num_screens) {
    LOG(WARNING)<< FormatStr("Selected screen number %d is outside the valid "
        "range [0,%d)",FLAGS_screen,num_screens);
    FLAGS_screen = 0;
  }
  LOG(INFO)<< FormatStr("Using X Screen number: (%d/%d)", FLAGS_screen,
      num_screens);
  g_root_win = XRootWindow(g_display, FLAGS_screen);
  XGetWindowAttributes(g_display, g_root_win, &g_root_attr);

  // Select events on root window. We install a special error handler so that
  // we can detect that another window manager has already selecte
  // SubstructureRedirect on the root window.
  g_wm_detected = false;
  XSetErrorHandler(&OnWMDetected);
  XSelectInput(g_display, g_root_win,
  SubstructureRedirectMask | SubstructureNotifyMask);

  // Force a sync so that we can know whether or not our XSelectInput call
  // has succeeded.
  XSync(g_display, false);
  if (g_wm_detected) {
    LOG(ERROR)<< "Detected another window manager on display "
    << XDisplayString(g_display);
    return 1;
  }

  // Now install our regular error handlers
  XSetErrorHandler(&OnXError);

  // Grab X server to prevent windows from changing under us.
  XGrabServer(g_display);

  // Reparent existing top-level windows
  Window returned_root, returned_parent;
  Window* top_level_windows;
  unsigned int num_top_level_windows;

  CHECK(
      XQueryTree(g_display, g_root_win, &returned_root, &returned_parent,
                 &top_level_windows, &num_top_level_windows));
  CHECK_EQ(returned_root, g_root_win);
  for (unsigned int i = 0; i < num_top_level_windows; ++i) {
    Frame(top_level_windows[i]);
  }
  XFree(top_level_windows);
  XUngrabServer(g_display);

  epoll_event epoll_spec;
  epoll_spec.events = EPOLLIN;
  epoll_spec.data.ptr = new EpollData { EPOLL_SIGNAL_FD, 0 };
  CHECK_NE(-1,
     epoll_ctl(g_epoll_fd, EPOLL_CTL_ADD, g_signal_read_fd, &epoll_spec))
     << "epoll_ctl: " << strerror(errno);

  epoll_spec.events = EPOLLIN | EPOLLPRI;
  epoll_spec.data.ptr = new EpollData { EPOLL_X_FD, 0 };
  CHECK_NE(-1,
      epoll_ctl(g_epoll_fd, EPOLL_CTL_ADD, XConnectionNumber(g_display),
                &epoll_spec))<< "epoll_ctl: " << strerror(errno);

  epoll_spec.events = EPOLLIN;
  epoll_spec.data.ptr = new EpollData { EPOLL_LISTEN_SOCKET_FD, 0 };
  CHECK_NE(-1,
      epoll_ctl(g_epoll_fd, EPOLL_CTL_ADD, g_listen_fd,
                &epoll_spec))<< "epoll_ctl: " << strerror(errno);


  std::array<epoll_event, 10> event_buf;

  PumpXQueue(); //< clear the queue before entering the poll loop
  while (!g_should_quit) {
    int epoll_return = epoll_wait(g_epoll_fd, event_buf.data(),
                                  event_buf.size(), 1000);
    if (epoll_return == -1) {
      if (errno = EINTR) {
        LOG(INFO)<< "Process was signalled";
      } else {
        LOG(FATAL)<< "epoll_wait : " << strerror(errno);
      }
    } else if(epoll_return == 0) {
      LOG(INFO) << "epoll timeout";
    } else {
      for (int i = 0; i < epoll_return; i++) {
        DispatchEpoll(event_buf[i]);
      }
    }
  }
  LOG(INFO) << "A clean exit";
  XCloseDisplay(g_display);
  return 0;
}
