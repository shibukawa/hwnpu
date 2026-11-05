package console

import "syscall"

const ioctlGetTermios, ioctlSetTermios = syscall.TIOCGETA, syscall.TIOCSETA
