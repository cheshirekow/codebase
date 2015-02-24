#ifndef TABWM_TREE_H_
#define TABWM_TREE_H_

#include <X11/Xlib.h>
#undef Success // defined by X.h, uh oh

#include <cstdint>
#include <array>

#include <Eigen/Dense>
#include <glog/logging.h>

namespace tabwm {

// bidirectional kd-tree node
struct Node {
  enum {
    HORIZONTAL,
    VERTICAL
  };

  Node() {
    split_type = HORIZONTAL;
    split_loc = 0;
    window = 0;
    parent = nullptr;
    children = {nullptr, nullptr};
  }

  int8_t split_type;
  int32_t split_loc;   //< in pixels
  Window window;
  Node* parent;
  std::array<Node*, 2> children;
};

inline bool IsInterior(const Node* node) {
  return node->children[0] != nullptr && node->children[1] != nullptr;
}

inline bool IsLeaf(const Node* node) {
  return node->children[0] == nullptr && node->children[1] == nullptr;
}

inline bool IsConsistent(const Node* node) {
  return IsInterior(node) || IsLeaf(node);
}

inline bool ContainsChild(const Node* parent, const Node* child) {
  return parent->children[0] == child || parent->children[1] == child;
}

inline Node* GetSibling(const Node* node) {
  const Node* parent = node->parent;
  DCHECK(ContainsChild(parent,node));
  return parent->children[0] == node ? parent->children[1]
                                     : parent->children[0];
}

inline void SplitNode(Node* node, Window w) {
  DCHECK(IsLeaf(node));
  Window sibling_w = node->window;
  node->window = 0;
  node->split_type = Node::HORIZONTAL;
  node->split_loc  = 0;

  Node* left_child = new Node();
  left_child->parent = node;
  left_child->window = sibling_w;

  Node* right_child = new Node();
  right_child->parent = node;
  right_child->window = w;

  node->children = {left_child, right_child};
}

/// Recursively search for the node at a specified location
inline Node* GetNodeAt(const Eigen::Vector2d& x, Node* node,
                       Eigen::Vector2d* origin, Eigen::Vector2d* size) {
  DCHECK(node);
  if(IsLeaf(node)) {
    return node;
  } else {
    if(node->split_type == Node::HORIZONTAL) {
      if(x[0] > (*origin)[0] + node->split_loc) {
        (*origin)[0] += node->split_loc;
        (*size)[0] -= node->split_loc;
        return GetNodeAt(x, node->children[1], origin, size);
      } else {
        (*size)[0] = node->split_loc;
        return GetNodeAt(x, node->children[0], origin, size);
      }
    } else {
      if(x[1] > (*origin)[1] + node->split_loc) {
        (*origin)[1] += node->split_loc;
        (*size)[1] -= node->split_loc;
        return GetNodeAt(x, node->children[1], origin, size);
      } else {
        (*size)[1] = node->split_loc;
        return GetNodeAt(x, node->children[0], origin, size);
      }
    }
  }
}

/// Generate a list of all windows by breadth first search
template <class OutputIterator>
void BreadthFirst(const Node* node, OutputIterator out) {
  if(!node) {
    return;
  }
  if(IsLeaf(node)) {
    *out++ = node->window;
  } else {
    BreadthFirst(node->children[0], out);
    BreadthFirst(node->children[1], out);
  }
}

class Tree {
public:
  Tree() :
      root_(nullptr) {}

  /// Remove a node from the tree
  /**
   *  @p node must be an leaf node. A node is only removed when the associated
   *  window is closed. When a node is removed it's parent is also removed,
   *  and it's sibling is promoted to the location previously occupied by
   *  the parent.
   *
   *  Does not destroy the node, the caller takes ownership of the pointer.
   */
  void RemoveNode(Node* node) {
    if(node == root_) {
      root_ = nullptr;
      return;
    }

    DCHECK(IsLeaf(node));
    Node* parent = node->parent;
    DCHECK(parent);
    DCHECK(ContainsChild(parent, node));

    if(parent == root_) {
      root_ = GetSibling(node);
      return;
    }

    Node* grand_parent = parent->parent;
    DCHECK(grand_parent);
    DCHECK(ContainsChild(grand_parent, parent));
    Node* sibling = GetSibling(node);
    DCHECK(sibling);
    if(grand_parent->children[0] == parent) {
      grand_parent->children[1] = sibling;
    } else {
      grand_parent->children[0] = sibling;
    }
    sibling->parent = grand_parent;
  }

  Node* GetNodeAt(const Eigen::Vector2d& x, Eigen::Vector2d* origin,
      Eigen::Vector2d* size) {
    if (root_) {
      *origin = {0, 0};
      return tabwm::GetNodeAt(x, root_, origin, size);
    } else {
      return nullptr;
    }
  }

  void InsertAt(const Eigen::Vector2d& x, Window w, Eigen::Vector2d* origin,
      Eigen::Vector2d* size) {
    if(root_) {
      *origin = {0,0};
      Node* parent = tabwm::GetNodeAt(x, root_, origin, size);
      SplitNode(parent, w);
      (*size)[0] /= 2.0;
      (*origin)[0] += (*size)[0];
      parent->split_loc = (*size)[0];
    } else {
      root_ = new Node;
      root_->window = w;
    }
  }

  template <class OutputIterator>
  void BreadthFirst(OutputIterator out) {
    tabwm::BreadthFirst(root_, out);
  }

private:
  Node* root_;
};

}  // namespace tabwm

#endif // TAWM_TREE_H_
