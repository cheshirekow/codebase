#include <sstream>
#include <gtest/gtest.h>

#include "../../src/tabwm/tree.h"

::testing::AssertionResult Vector2dIs(const Eigen::Vector2d& expected_size,
      const Eigen::Vector2d& actual_size) {
    for (int i = 0; i < 2; i++) {
      if (actual_size[i] != expected_size[i]) {
        return ::testing::AssertionFailure()
            << "Vector2d does not match expected"
            << "\n expected: " << expected_size.transpose()
            << "\n actual:   " << actual_size.transpose() << "\n";
      }
    }
    return ::testing::AssertionSuccess();
  }

template <class Container>
std::string FormatWindowList(const Container& windows) {
  std::stringstream result;
  result << "[";
  if(windows.size() < 1) {
    result << "empty";
  } else if(windows.size() < 2) {
    result << windows.front();
  } else {
    auto end = windows.end();
    auto last = end;
    last--;
    for(auto iter = windows.begin(); iter != last; ++iter) {
      result << *iter << ", ";
    }
    result << *last;
  }
  result << "]";
  return result.str();
}

class TreeTest : public ::testing::Test {
 protected:
  tabwm::Tree tree_;

 public:
  ::testing::AssertionResult InOrderTreeIs(
      const std::vector<Window>& windows_in_order) {
    std::vector<Window> expected = windows_in_order;
    std::vector<Window> actual;
    tree_.BreadthFirst(std::back_inserter(actual));
    if(expected == actual) {
      return ::testing::AssertionSuccess();
    } else {
      return ::testing::AssertionFailure()
      << "In-order tree-traversal does not match expected:"
          << "\n expected: " << FormatWindowList(expected)
          << "\n   actual: " << FormatWindowList(actual)
          << "\n";
    }
  }
};

TEST_F(TreeTest, CreateTreeByInsertionAtLocation) {
  const Eigen::Vector2d screen_size(800,600);
  Eigen::Vector2d origin(0,0);
  Eigen::Vector2d window_size = screen_size;

  tree_.InsertAt({400,300}, 0, &origin, &window_size);
  EXPECT_TRUE(Vector2dIs({0,0}, origin));
  EXPECT_TRUE(Vector2dIs({800,600}, window_size));
  EXPECT_TRUE(InOrderTreeIs({0}));

  window_size = screen_size;
  tree_.InsertAt({400,300}, 1, &origin, &window_size);
  EXPECT_TRUE(Vector2dIs({400,0}, origin));
  EXPECT_TRUE(Vector2dIs({400,600}, window_size));
  EXPECT_TRUE(InOrderTreeIs({0,1}));

  window_size = screen_size;
  tree_.GetNodeAt({600,300}, &origin, &window_size);
  EXPECT_TRUE(Vector2dIs({400,0}, origin));
  EXPECT_TRUE(Vector2dIs({400,600}, window_size));

  window_size = screen_size;
  tree_.InsertAt({600,300}, 2, &origin, &window_size);
  EXPECT_TRUE(Vector2dIs({600,0}, origin));
  EXPECT_TRUE(Vector2dIs({200,600}, window_size));
  EXPECT_TRUE(InOrderTreeIs({0,1,2}));
}

