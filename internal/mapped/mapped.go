// Package mapped gives a whole file's bytes mapped into memory, read only: the system reads the
// pages as they are touched and may drop them again, so a table of hundreds of MB read a row at a
// time takes neither the time to read it all nor that much of the process's memory. Where there
// is no mapping (Windows), the file is read.
package mapped

// File is a file's bytes, valid until Close.
type File struct {
	b      []byte
	mapped bool
}

// Bytes are the file's bytes. They must not be written, and not be used after Close.
func (f *File) Bytes() []byte { return f.b }

// Len is the file's size.
func (f *File) Len() int { return len(f.b) }

// Close lets the bytes go; a nil File or one closed already is left alone.
func (f *File) Close() error {
	if f == nil || f.b == nil {
		return nil
	}
	b := f.b
	f.b = nil
	if f.mapped {
		return unmap(b)
	}
	return nil
}
