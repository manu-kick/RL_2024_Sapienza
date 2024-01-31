#ifndef ORDER_H
#define ORDER_H

// Define the orders that can be sent and received
enum Order {
  HELLO = 0,
  TOUCH_L = 1,
  TOUCH_R = 2,
  MOTOR = 3,
  ALREADY_CONNECTED = 4,
  ERROR = 5,
  RECEIVED = 6,
  STOP = 7,
};

typedef enum Order Order;

#endif