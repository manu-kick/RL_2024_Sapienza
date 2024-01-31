#ifndef PARAMETERS_H
#define PARAMETERS_H

#define SERIAL_BAUD 115200  // Baudrate

// L298N pins left
#define en_l 11
#define in1_l 10
#define in2_l 9

// L298N pins right
#define en_r 8
#define in1_r 5
#define in2_r 4

#define yl99_r_pin 13
#define yl99_l_pin 12

#define MOTORS_SPEED 200

// #define SERVOMOTOR_PIN 6
// #define INITIAL_THETA 110  // Initial angle of the servomotor
// // Min and max values for motors
// #define THETA_MIN 60
// #define THETA_MAX 150

#define SPEED_MAX 100

// If DEBUG is set to true, the arduino will send back all the received messages
#define DEBUG true

#endif