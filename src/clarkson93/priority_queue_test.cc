/*
 * */
#include <gtest/gtest.h>
#include "clarkson93/priority_queue.h"

TEST(PriorityQueueTest, RangeForTest) {
  clarkson93::PriorityQueue<int> pqueue;

  for (int value : {1, 9, 3, 5, 2, 4, 6, 7, 0, 8}) {
    pqueue.push(value);
  }

  // test iteration over elements. They will be in heap order so we just need to
  // check that they're all there.
  bool value_in_queue[10];
  for (int value : pqueue) {
    ASSERT_LT(-1, value);
    ASSERT_LT(value, 10);
    value_in_queue[value] = true;
  }
  for (int i = 0; i < 10; i++) {
    EXPECT_TRUE(value_in_queue[i]) << "For element " << i;
  }
}

TEST(PriorityQueueTest, InOrderPopTest) {
  clarkson93::PriorityQueue<int> pqueue;

  for (int value : {1, 9, 3, 5, 2, 4, 6, 7, 0, 8}) {
    pqueue.push(value);
  }

  for (int i = 0; i < 10; i++) {
    EXPECT_EQ(i, pqueue.Pop());
  }
}

TEST(PriorityQueueTest, ReserveTest) {
  clarkson93::PriorityQueue<int> pqueue;
  pqueue.Reserve(100);

  const int* first_element_pointer_before_insertion = &(*pqueue.begin());
  for (int i = 0; i < 100; i++) {
    pqueue.push(i);
  }

  const int* first_element_pointer_after_insertion = &(*pqueue.begin());
  EXPECT_EQ(first_element_pointer_before_insertion,
            first_element_pointer_after_insertion);
}

TEST(PriorityQueueTest, ClearTest) {
  clarkson93::PriorityQueue<int> pqueue;
  for (int value : {1, 4, 9, 10}) {
    pqueue.push(value);
  }
  EXPECT_NE(pqueue.begin(), pqueue.end());
  pqueue.Clear();
  EXPECT_EQ(pqueue.begin(), pqueue.end());
}
