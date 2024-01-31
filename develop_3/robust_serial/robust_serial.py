import struct
from enum import Enum
from typing import BinaryIO


class Order(Enum):
    """
    Pre-defined orders
    """
    
    HELLO = 0
    TOUCH_L = 1
    TOUCH_R = 2
    MOTOR = 3
    ALREADY_CONNECTED = 4
    ERROR = 5
    RECEIVED = 6
    STOP = 7


def read_order(f: BinaryIO) -> Order:
    """
    :param f: file handler or serial file
    :return: (Order Enum Object)
    """
    return Order(read_i8(f))


def read_i8(f: BinaryIO) -> Order:
    """
    :param f: file handler or serial file
    :return: (int8_t)
    """
    return struct.unpack("<b", bytearray(f.read(1)))[0]


def read_i16(f: BinaryIO) -> Order:
    """
    :param f: file handler or serial file
    :return: (int16_t)
    """
    return struct.unpack("<h", bytearray(f.read(2)))[0]


def read_i32(f):
    """
    :param f: file handler or serial file
    :return: (int32_t)
    """
    return struct.unpack("<l", bytearray(f.read(4)))[0]


def write_i8(f: BinaryIO, value: int) -> None:
    """
    :param f: file handler or serial file
    :param value: (int8_t)
    """
    if -128 <= value <= 127:
        f.write(struct.pack("<b", value))
    else:
        print(f"Value error:{value}")


def write_order(f: BinaryIO, order: Order) -> None:
    """
    :param f: file handler or serial file
    :param order: (Order Enum Object)
    """
    write_i8(f, order.value)


def write_i16(f: BinaryIO, value: int) -> None:
    """
    :param f: file handler or serial file
    :param value: (int16_t)
    """
    f.write(struct.pack("<h", value))


def write_i32(f: BinaryIO, value: int) -> None:
    """
    :param f: file handler or serial file
    :param value: (int32_t)
    """
    f.write(struct.pack("<l", value))


def decode_order(f: BinaryIO, byte: int, debug: bool = False) -> None:
    """
    :param f: file handler or serial file
    :param byte: (int8_t)
    :param debug: (bool) whether to print or not received messages
    """
    try:
        order = Order(byte)
        if order == Order.HELLO:
            msg = f"HELLO {byte}"
        elif order == Order.TOUCH_L:
            # touch_l = read_i8(f)
            # msg = f"TOUCH_L {touch_l}"
            msg = f"TOUCH_L"
        elif order == Order.TOUCH_R:
            touch_r = read_i8(f)
            msg = f"TOUCH_R {touch_r}"
        elif order == Order.MOTOR:
            action = read_i8(f)
            msg = f"motor action {action}"
        elif order == Order.ALREADY_CONNECTED:
            msg = "ALREADY_CONNECTED"
        elif order == Order.ERROR:
            error_code = read_i16(f)
            msg = f"Error {error_code}"
        elif order == Order.RECEIVED:
            msg = "RECEIVED"
        elif order == Order.STOP:
            msg = "STOP"
        else:
            msg = ""
            print("Unknown Order", byte)

        if debug:
            print(msg)
    except Exception as e:
        print(f"Error decoding order {byte}: {e}")
        print(f"byte={byte:08b}")
