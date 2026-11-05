"""Print the NPU clocks from GET_INFO(QUERY_CLOCK_METADATA)."""
import ctypes, fcntl, os, struct

buf = ctypes.create_string_buffer(48)
arg = struct.pack("<IIQ", 3, 48, ctypes.addressof(buf))  # param, buffer_size, buffer
req = (3 << 30) | (16 << 16) | (ord("d") << 8) | (0x40 + 7)
fd = os.open("/dev/accel/accel0", os.O_RDWR)
fcntl.ioctl(fd, req, bytearray(arg))
os.close(fd)
for i in range(2):
    name, mhz = struct.unpack_from("<16sI", buf.raw, i * 24)
    print(name.rstrip(b"\0").decode(), mhz, "MHz")
