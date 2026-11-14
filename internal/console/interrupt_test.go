//go:build unix

package console

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"slices"
	"strings"
	"syscall"
	"testing"
	"time"
)

// TestInterruptHelper is the command the tests below run and interrupt, in the way
// HWNPU_INTERRUPT_HELPER names.
func TestInterruptHelper(t *testing.T) {
	mode := os.Getenv("HWNPU_INTERRUPT_HELPER")
	if mode == "" {
		t.Skip("the helper of the interrupt tests")
	}
	if mode == "twice" {
		grace = time.Minute
	}
	ctx, stop := Interrupt()
	fmt.Println("ready")
	switch mode {
	case "busy": // a command on the NPU: the context is not looked at
		time.Sleep(time.Minute)
	case "returns": // what watches the context: it returns, with the cancellation as its error
		<-ctx.Done()
		fmt.Println("returned")
		EndInterrupted(ctx.Err())
		fmt.Println("still here")
	case "twice": // the context ends, the command does not, and the grace is a minute
		<-ctx.Done()
		fmt.Println("canceled")
		time.Sleep(time.Minute)
	case "stopped": // a chat: Ctrl-C is its own once the model is loaded
		stop()
		sig := make(chan os.Signal, 1)
		signal.Notify(sig, os.Interrupt)
		fmt.Println("stopped")
		<-sig
		time.Sleep(3 * InterruptGrace)
		EndInterrupted(errors.New("an error of its own"))
		fmt.Println("alive")
	}
	os.Exit(0)
}

// interrupt runs the helper in mode, sends it SIGINT each time it has printed one of the lines
// at, and returns what it printed, how it ended and how long that took from the last signal.
func interrupt(t *testing.T, mode string, at ...string) (out string, err error, took time.Duration) {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=^TestInterruptHelper$")
	cmd.Env = append(os.Environ(), "HWNPU_INTERRUPT_HELPER="+mode)
	pipe, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	var b strings.Builder
	sc := bufio.NewScanner(pipe)
	var t0 time.Time
	for sc.Scan() {
		b.WriteString(sc.Text() + "\n")
		if slices.Contains(at, sc.Text()) {
			t0 = time.Now()
			cmd.Process.Signal(syscall.SIGINT)
		}
	}
	err = cmd.Wait()
	return b.String(), err, time.Since(t0)
}

// killedBySIGINT reports whether err is of a process the interrupt signal ended.
func killedBySIGINT(err error) bool {
	var exit *exec.ExitError
	if !errors.As(err, &exit) {
		return false
	}
	ws, ok := exit.Sys().(syscall.WaitStatus)
	return ok && ws.Signaled() && ws.Signal() == syscall.SIGINT
}

func TestInterruptEndsABusyCommand(t *testing.T) {
	out, err, took := interrupt(t, "busy", "ready")
	if !killedBySIGINT(err) {
		t.Fatalf("ended with %v, not by SIGINT (printed %q)", err, out)
	}
	if took < InterruptGrace || took > InterruptGrace+2*time.Second {
		t.Errorf("ended %v after the signal: want the grace of %v", took, InterruptGrace)
	}
}

func TestInterruptEndsACommandThatReturns(t *testing.T) {
	out, err, _ := interrupt(t, "returns", "ready")
	if !killedBySIGINT(err) {
		t.Fatalf("ended with %v, not by SIGINT (printed %q)", err, out)
	}
	if !strings.Contains(out, "returned") || strings.Contains(out, "still here") {
		t.Errorf("printed %q: want the command to return, and EndInterrupted to end the process", out)
	}
}

func TestInterruptTwiceEndsAtOnce(t *testing.T) {
	out, err, took := interrupt(t, "twice", "ready", "canceled")
	if !killedBySIGINT(err) {
		t.Fatalf("ended with %v, not by SIGINT (printed %q)", err, out)
	}
	if took > 10*time.Second {
		t.Errorf("ended %v after the second signal: it waited for the grace", took)
	}
}

func TestInterruptStopped(t *testing.T) {
	out, err, _ := interrupt(t, "stopped", "stopped")
	if err != nil {
		t.Fatalf("ended with %v (printed %q): want the command's own handling after stop", err, out)
	}
	if !strings.Contains(out, "alive") {
		t.Errorf("printed %q", out)
	}
}
