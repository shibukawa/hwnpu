//go:build linux || darwin

package console

import (
	"os"
	"syscall"
	"unsafe"
)

// makeRaw puts the terminal fd in raw mode, as cfmakeraw does but keeping the output's line ends
// (OPOST), and returns what restores it.
func makeRaw(fd int) (restore func(), err error) {
	var old syscall.Termios
	if err := ioctl(fd, ioctlGetTermios, unsafe.Pointer(&old)); err != nil {
		return nil, err
	}
	t := old
	t.Iflag &^= syscall.IGNBRK | syscall.BRKINT | syscall.PARMRK | syscall.ISTRIP | syscall.INLCR | syscall.IGNCR | syscall.ICRNL | syscall.IXON
	t.Lflag &^= syscall.ECHO | syscall.ECHONL | syscall.ICANON | syscall.ISIG | syscall.IEXTEN
	t.Cflag &^= syscall.CSIZE | syscall.PARENB
	t.Cflag |= syscall.CS8
	t.Cc[syscall.VMIN], t.Cc[syscall.VTIME] = 1, 0
	if err := ioctl(fd, ioctlSetTermios, unsafe.Pointer(&t)); err != nil {
		return nil, err
	}
	return func() { ioctl(fd, ioctlSetTermios, unsafe.Pointer(&old)) }, nil
}

// columns is the width of the terminal f, 80 when it cannot be read.
func columns(f *os.File) int {
	var ws struct{ Row, Col, X, Y uint16 }
	if err := ioctl(int(f.Fd()), syscall.TIOCGWINSZ, unsafe.Pointer(&ws)); err != nil || ws.Col == 0 {
		return 80
	}
	return int(ws.Col)
}

func ioctl(fd int, req uintptr, arg unsafe.Pointer) error {
	if _, _, e := syscall.Syscall(syscall.SYS_IOCTL, uintptr(fd), req, uintptr(arg)); e != 0 {
		return e
	}
	return nil
}
