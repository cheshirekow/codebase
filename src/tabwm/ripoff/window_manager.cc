#include "window_manager.h"

#include <sys/select.h>
#include <algorithm>
#include <cstring>
#include <set>
#include <glog/logging.h>
#include "util.h"

// Whether an existing window manager has been detected. Set by OnWMDetected.
bool g_wm_detected;

// A mutex for protecting wm_detected_.
std::mutex g_wm_detected_mutex;

// Xlib error handler.
int OnXError(Display* display, XErrorEvent* e);

// Xlib error handler used to determine whether another window manager is
// running. It is set as the error handler right before selecting substructure
// redirection mask on the root window, so it is invoked223 if and only if
// another window manager is running.
int OnWMDetected(Display* display, XErrorEvent* e);

std::unique_ptr<WindowManager> WindowManager::Create(
    const std::string& display_str, int signal_fd) {
  // 1. Open X display.
  const char* display_c_str =
      display_str.empty() ? nullptr : display_str.c_str();
  LOG(INFO) << "Opening X display " << display_str;
  Display* display = XOpenDisplay(display_c_str);
  if (display == nullptr) {
    LOG(ERROR)<< "Failed to open X display " << XDisplayName(display_c_str);
    return nullptr;
  }
  // 2. Construct WindowManager instance.
  return std::unique_ptr<WindowManager>(new WindowManager(display, signal_fd));
}

WindowManager::WindowManager(Display* display, int signal_fd) {
  signal_fd_ = signal_fd;
  display_ = CHECK_NOTNULL(display);
  WM_PROTOCOLS = XInternAtom(display_, "WM_PROTOCOLS", false);
  WM_DELETE_WINDOW = XInternAtom(display_, "WM_DELETE_WINDOW", false);
}

WindowManager::~WindowManager() {
  XCloseDisplay(display_);
}

void WindowManager::Run() {
  // In the future, we will not need this
  root_ = DefaultRootWindow(display_);

  // Get the window attributes for the root window on each screen
  int num_screens = ScreenCount(display_);
  root_attr_.resize(num_screens);
  for (int i = 0; i < num_screens; i++) {
    Window root_window = XRootWindow(display_, i);
    XGetWindowAttributes(display_, root_window, &root_attr_[i]);
  }

  // 1. Initialization.
  //   a. Select events on root window. Use a special error handler so we can
  //   exit gracefully if another window manager is already running.
  {
    std::lock_guard<std::mutex> lock(g_wm_detected_mutex);

    g_wm_detected = false;
    XSetErrorHandler(&OnWMDetected);
    for (const auto& win_attr : root_attr_) {
      XSelectInput(display_, win_attr.root,
      SubstructureRedirectMask | SubstructureNotifyMask);
    }

    XSync(display_, false);
    if (g_wm_detected) {
      LOG(ERROR) << "Detected another window manager on display "
                 << XDisplayString(display_);
      return;
    }
  }
  //   b. Set error handler.
  XSetErrorHandler(&OnXError);
  //   c. Grab X server to prevent windows from changing under us.
  XGrabServer(display_);
  //   d. Reparent existing top-level windows.
  //     i. Query existing top-level windows.
  Window returned_root, returned_parent;
  Window* top_level_windows;
  unsigned int num_top_level_windows;
  CHECK(XQueryTree(
      display_,
      root_,
      &returned_root,
      &returned_parent,
      &top_level_windows,
      &num_top_level_windows));
  CHECK_EQ(returned_root, root_);
  //     ii. Frame each top-level window.
  for (unsigned int i = 0; i < num_top_level_windows; ++i) {
    Frame(top_level_windows[i]);
  }
  //     iii. Free top-level window array.
  XFree(top_level_windows);
  //   e. Ungrab X server.
  XUngrabServer(display_);

  std::vector<int> file_descriptors(2);
  file_descriptors[0] = signal_fd_;
  file_descriptors[1] = XConnectionNumber(display_);
  std::make_heap(file_descriptors.begin(), file_descriptors.end());
  int n_fd = file_descriptors[0] + 1;
  bool should_quit = false;
  fd_set select_me;

  while(!should_quit) {
    // select on the X and signal file descriptors
    FD_ZERO(&select_me);
    for (int fd : file_descriptors) {
      FD_SET(fd, &select_me);
    }
    timeval timeout{1,0};
    int n_ready = select(n_fd, &select_me, nullptr, nullptr, &timeout);

    if(FD_ISSET(signal_fd_, &select_me)) {
      should_quit = true;
    }

    while (XPending(display_)) {
      // 1. Get next event.
      XEvent e;
      XNextEvent(display_, &e);
      LOG(INFO)<< "Received event: " << XEventToString(e);

      // 2. Dispatch event.
      switch (e.type) {
      case CreateNotify:
        OnCreateNotify(e.xcreatewindow);
        break;
      case DestroyNotify:
        OnDestroyNotify(e.xdestroywindow);
        break;
      case ReparentNotify:
        OnReparentNotify(e.xreparent);
        break;
      case MapNotify:
        OnMapNotify(e.xmap);
        break;
      case UnmapNotify:
        OnUnmapNotify(e.xunmap);
        break;
      case ConfigureNotify:
        OnConfigureNotify(e.xconfigure);
        break;
      case MapRequest:
        OnMapRequest(e.xmaprequest);
        break;
      case ConfigureRequest:
        OnConfigureRequest(e.xconfigurerequest);
        break;
      case ButtonPress:
        OnButtonPress(e.xbutton);
        break;
      case ButtonRelease:
        OnButtonRelease(e.xbutton);
        break;
      case MotionNotify:
        // Skip any already pending motion events.
        while (XCheckTypedWindowEvent(display_, e.xmotion.window, MotionNotify,
            &e)) {
        }
        OnMotionNotify(e.xmotion);
        break;
      case KeyPress:
        OnKeyPress(e.xkey);
        break;
      case KeyRelease:
        OnKeyRelease(e.xkey);
        break;
      default:
        LOG(WARNING)<< "Ignored event";
      }
    }
  }
}

void WindowManager::Frame(Window w) {
  // Visual properties of the frame to create.
  const unsigned int BORDER_WIDTH = 3;
  const unsigned long BORDER_COLOR = 0xff0000;
  const unsigned long BG_COLOR = 0x0000ff;

  CHECK(!clients_.count(w));

  // 1. Retrieve attributes of window to frame.
  XWindowAttributes x_window_attrs;
  CHECK(XGetWindowAttributes(display_, w, &x_window_attrs));
  // 2. Create frame.
  const Window frame = XCreateSimpleWindow(
      display_,
      root_,
      x_window_attrs.x,
      x_window_attrs.y,
      x_window_attrs.width,
      x_window_attrs.height,
      BORDER_WIDTH,
      BORDER_COLOR,
      BG_COLOR);
  // 3. Select events on frame.
  XSelectInput(
      display_,
      frame,
      SubstructureRedirectMask | SubstructureNotifyMask);
  // 4. Add client to save set, so that it will be restored and kept alive if we
  // crash.
  XAddToSaveSet(display_, w);
  // 5. Reparent client window.
  XReparentWindow(
      display_,
      w,
      frame,
      0, 0);  // Offset of client window within frame.
  // 6. Map frame.
  XMapWindow(display_, frame);
  // 7. Save frame handle.
  clients_[w] = frame;
  // 8. Grab universal window management actions on client window.
  //   a. Move windows with alt + left button.
  XGrabButton(
      display_,
      Button1,
      Mod1Mask,
      w,
      false,
      ButtonPressMask | ButtonReleaseMask | ButtonMotionMask,
      GrabModeAsync,
      GrabModeAsync,
      None,
      None);
  //   b. Resize windows with alt + right button.
  XGrabButton(
      display_,
      Button3,
      Mod1Mask,
      w,
      false,
      ButtonPressMask | ButtonReleaseMask | ButtonMotionMask,
      GrabModeAsync,
      GrabModeAsync,
      None,
      None);
  //   c. Kill windows with alt + f4.
  XGrabKey(
      display_,
      XKeysymToKeycode(display_, XK_F4),
      Mod1Mask,
      w,
      false,
      GrabModeAsync,
      GrabModeAsync);
  //   d. Switch windows with alt + tab.
  XGrabKey(
      display_,
      XKeysymToKeycode(display_, XK_Tab),
      Mod1Mask,
      w,
      false,
      GrabModeAsync,
      GrabModeAsync);

  LOG(INFO) << "Framed window " << w << " [" << frame << "]";
}

void WindowManager::Unframe(Window w) {
  CHECK(clients_.count(w));

  // We reverse the steps taken in Frame().
  const Window frame = clients_[w];
  // 1. Unmap frame.
  XUnmapWindow(display_, frame);
  // 2. Reparent client window.
  XReparentWindow(
      display_,
      w,
      root_,
      0, 0);  // Offset of client window within root.
  // 3. Remove client window from save set, as it is now unrelated to us.
  XRemoveFromSaveSet(display_, w);
  // 4. Destroy frame.
  XDestroyWindow(display_, frame);
  // 5. Drop reference to frame handle.
  clients_.erase(w);

  LOG(INFO) << "Unframed window " << w << " [" << frame << "]";
}

void WindowManager::OnCreateNotify(const XCreateWindowEvent& e) {}

void WindowManager::OnDestroyNotify(const XDestroyWindowEvent& e) {}

void WindowManager::OnReparentNotify(const XReparentEvent& e) {}

void WindowManager::OnMapNotify(const XMapEvent& e) {}

void WindowManager::OnUnmapNotify(const XUnmapEvent& e) {
  // If the window is a client window we manage, unframe it upon UnmapNotify. We
  // need the check because other than a client window, we can receive an
  // UnmapNotify for
  //     - A frame we just destroyed ourselves.
  //     - A pre-existing and mapped top-level window we reparented.
  if (!clients_.count(e.window)) {
    LOG(INFO) << "Ignore UnmapNotify for non-client window " << e.window;
    return;
  }
  if (e.event == root_) {
    LOG(INFO) << "Ignore UnmapNotify for reparented pre-existing window "
              << e.window;
    return;
  }
  Unframe(e.window);
}

void WindowManager::OnConfigureNotify(const XConfigureEvent& e) {}

void WindowManager::OnMapRequest(const XMapRequestEvent& e) {
  // 1. Frame or re-frame window.
  Frame(e.window);
  // 2. Actually map window.
  XMapWindow(display_, e.window);
}

void WindowManager::OnConfigureRequest(const XConfigureRequestEvent& e) {
  XWindowChanges changes;
  changes.x = e.x;
  changes.y = e.y;
  changes.width = e.width;
  changes.height = e.height;
  changes.border_width = e.border_width;
  changes.sibling = e.above;
  changes.stack_mode = e.detail;
  if (clients_.count(e.window)) {
    const Window frame = clients_[e.window];
    XConfigureWindow(display_, frame, e.value_mask, &changes);
    LOG(INFO)<< "Resize [" << frame << "] to " << Eigen::Vector2i(e.width, e.height);
  }
  XConfigureWindow(display_, e.window, e.value_mask, &changes);
  LOG(INFO)<< "Resize " << e.window << " to " << Eigen::Vector2i(e.width, e.height);
}

void WindowManager::OnButtonPress(const XButtonEvent& e) {
  CHECK(clients_.count(e.window));
  const Window frame = clients_[e.window];

  // 1. Save initial cursor position.
  drag_start_pos_ = Eigen::Vector2i(e.x_root, e.y_root);

  // 2. Save initial window info.
  Window returned_root;
  int x, y;
  unsigned width, height, border_width, depth;
  CHECK(XGetGeometry(
      display_,
      frame,
      &returned_root,
      &x, &y,
      &width, &height,
      &border_width,
      &depth));
  drag_start_frame_pos_ = Eigen::Vector2i(x, y);
  drag_start_frame_size_ = Eigen::Vector2i(width, height);

  // 3. Raise clicked window to top.
  XRaiseWindow(display_, frame);
}

void WindowManager::OnButtonRelease(const XButtonEvent& e) {}

void WindowManager::OnMotionNotify(const XMotionEvent& e) {
  CHECK(clients_.count(e.window));
  const Window frame = clients_[e.window];
  const Eigen::Vector2i drag_pos(e.x_root, e.y_root);
  const Eigen::Vector2i delta = drag_pos - drag_start_pos_;

  if (e.state & Button1Mask ) {
    // alt + left button: Move window.
    const Eigen::Vector2i dest_frame_pos = drag_start_frame_pos_ + delta;
    XMoveWindow(
        display_,
        frame,
        dest_frame_pos[0], dest_frame_pos[1]);
  } else if (e.state & Button3Mask) {
    // alt + right button: Resize window.
    // Window dimensions cannot be negative.
    const Eigen::Vector2i size_delta(
        std::max(delta[0], -drag_start_frame_size_[0]),
        std::max(delta[1], -drag_start_frame_size_[1]));
    const Eigen::Vector2i dest_frame_size = drag_start_frame_size_ + size_delta;
    // 1. Resize frame.
    XResizeWindow(
        display_,
        frame,
        dest_frame_size[0], dest_frame_size[1]);
    // 2. Resize client window.
    XResizeWindow(
        display_,
        e.window,
        dest_frame_size[0], dest_frame_size[1]);
  }
}

void WindowManager::OnKeyPress(const XKeyEvent& e) {
  if ((e.state & Mod1Mask) &&
      (e.keycode == XKeysymToKeycode(display_, XK_F4))) {
    // alt + f4: Close window.
    //
    // There are two ways to tell an X window to close. The first is to send it
    // a message of type WM_PROTOCOLS and value WM_DELETE_WINDOW. If the client
    // has not explicitly marked itself as supporting this more civilized
    // behavior (using XSetWMProtocols()), we kill it with XKillClient().
    Atom* supported_protocols;
    int num_supported_protocols;
    if (XGetWMProtocols(display_,
                        e.window,
                        &supported_protocols,
                        &num_supported_protocols) &&
        (std::find(supported_protocols,
                   supported_protocols + num_supported_protocols,
                   WM_DELETE_WINDOW) !=
         supported_protocols + num_supported_protocols)) {
      LOG(INFO) << "Gracefully deleting window " << e.window;
      // 1. Construct message.
      XEvent msg;
      memset(&msg, 0, sizeof(msg));
      msg.xclient.type = ClientMessage;
      msg.xclient.message_type = WM_PROTOCOLS;
      msg.xclient.window = e.window;
      msg.xclient.format = 32;
      msg.xclient.data.l[0] = WM_DELETE_WINDOW;
      // 2. Send message to window to be closed.
      CHECK(XSendEvent(display_, e.window, false, 0, &msg));
    } else {
      LOG(INFO) << "Killing window " << e.window;
      XKillClient(display_, e.window);
    }
  } else if ((e.state & Mod1Mask) &&
             (e.keycode == XKeysymToKeycode(display_, XK_Tab))) {
    // alt + tab: Switch window.
    // 1. Find next window.
    auto i = clients_.find(e.window);
    CHECK(i != clients_.end());
    ++i;
    if (i == clients_.end()) {
      i = clients_.begin();
    }
    // 2. Raise and set focus.
    XRaiseWindow(display_, i->second);
    XSetInputFocus(display_, i->first, RevertToPointerRoot, CurrentTime);
  }
}

void WindowManager::OnKeyRelease(const XKeyEvent& e) {}

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
