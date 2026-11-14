//go:build darwin

package npuinfo

import (
	"context"
	"errors"
	"runtime"
	"sync"
	"time"
	"unsafe"
)

// IOReport is the private library (/usr/lib/libIOReport.dylib) behind powermetrics and
// Activity Monitor's energy figures. Unlike powermetrics it needs no root. Its functions are
// not documented; the signatures here are the ones open-source monitors (asitop, macmon,
// socpowerbud) use, and what darwin_test checks on the machine it runs on.

const kCFStringEncodingUTF8 = 0x08000100

// IOReport channel formats (IOReportTypes.h).
const (
	iorFormatSimple    = 1
	iorFormatState     = 2
	iorFormatHistogram = 3
	iorFormatSimpleArr = 4
)

type cfLib struct {
	StringCreateWithCString, StringGetCString, Release, DictionaryGetValue, DictionaryGetCount,
	DictionaryCreateMutableCopy, ArrayGetCount, ArrayGetValueAtIndex uintptr
}

type iorLib struct {
	cf                                                                          cfLib
	CopyChannelsInGroup, MergeChannels, CreateSubscription, CreateSamples       uintptr
	CreateSamplesDelta, ChannelGetGroup, ChannelGetSubGroup, ChannelGetName     uintptr
	ChannelGetUnitLabel, ChannelGetFormat, SimpleGetIntegerValue, StateGetCount uintptr
	StateGetNameForIndex, StateGetResidency                                     uintptr
}

var loadIOReport = sync.OnceValues(func() (*iorLib, error) {
	cf, err := dlopen("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
	if err != nil {
		return nil, err
	}
	ior, err := dlopen("/usr/lib/libIOReport.dylib")
	if err != nil {
		return nil, err
	}
	l := &iorLib{}
	c := &l.cf
	if err := cf.syms([]string{"CFStringCreateWithCString", "CFStringGetCString", "CFRelease",
		"CFDictionaryGetValue", "CFDictionaryGetCount", "CFDictionaryCreateMutableCopy",
		"CFArrayGetCount", "CFArrayGetValueAtIndex"},
		&c.StringCreateWithCString, &c.StringGetCString, &c.Release, &c.DictionaryGetValue,
		&c.DictionaryGetCount, &c.DictionaryCreateMutableCopy, &c.ArrayGetCount,
		&c.ArrayGetValueAtIndex); err != nil {
		return nil, err
	}
	if err := ior.syms([]string{"IOReportCopyChannelsInGroup", "IOReportMergeChannels",
		"IOReportCreateSubscription", "IOReportCreateSamples", "IOReportCreateSamplesDelta",
		"IOReportChannelGetGroup", "IOReportChannelGetSubGroup", "IOReportChannelGetChannelName",
		"IOReportChannelGetUnitLabel", "IOReportChannelGetFormat", "IOReportSimpleGetIntegerValue",
		"IOReportStateGetCount", "IOReportStateGetNameForIndex", "IOReportStateGetResidency"},
		&l.CopyChannelsInGroup, &l.MergeChannels, &l.CreateSubscription, &l.CreateSamples,
		&l.CreateSamplesDelta, &l.ChannelGetGroup, &l.ChannelGetSubGroup, &l.ChannelGetName,
		&l.ChannelGetUnitLabel, &l.ChannelGetFormat, &l.SimpleGetIntegerValue, &l.StateGetCount,
		&l.StateGetNameForIndex, &l.StateGetResidency); err != nil {
		return nil, err
	}
	return l, nil
})

// cfString makes a CFString; the caller releases it.
func (c *cfLib) cfString(s string) uintptr {
	p := cstring(s)
	r := ccall(c.StringCreateWithCString, 0, uintptr(unsafe.Pointer(p)), kCFStringEncodingUTF8)
	runtime.KeepAlive(p)
	return r
}

func (c *cfLib) goString(s uintptr) string {
	if s == 0 {
		return ""
	}
	buf := make([]byte, 1024)
	if ccall(c.StringGetCString, s, uintptr(unsafe.Pointer(&buf[0])), uintptr(len(buf)), kCFStringEncodingUTF8) == 0 {
		return ""
	}
	return gostringN(buf)
}

func (c *cfLib) release(p uintptr) {
	if p != 0 {
		ccall(c.Release, p)
	}
}

// aneIOReportGroups are the (group, subgroup) pairs subscribed to; "" is every subgroup.
// Only the channels isANEChannel accepts are kept. Groups a chip does not have are skipped.
var aneIOReportGroups = [][2]string{
	{"Energy Model", ""},
	{"PMP", ""},
	{"AMC Stats", "Perf Counters"},
	{"SoC Stats", "Events"},
	{"Interrupt Statistics (by index)", ""},
}

// darwinIOReport samples the ANE's IOReport channels twice, interval apart, and returns the
// differences. Reading the counters does not touch the ANE.
func darwinIOReport(ctx context.Context, interval time.Duration) (*DarwinIOReport, error) {
	l, err := loadIOReport()
	if err != nil {
		return nil, err
	}
	c := &l.cf
	var chans uintptr
	for _, g := range aneIOReportGroups {
		gs := c.cfString(g[0])
		var ss uintptr
		if g[1] != "" {
			ss = c.cfString(g[1])
		}
		d := ccall(l.CopyChannelsInGroup, gs, ss, 0, 0, 0)
		c.release(gs)
		c.release(ss)
		if d == 0 {
			continue
		}
		if chans == 0 {
			chans = d
			continue
		}
		ccall(l.MergeChannels, chans, d, 0)
		c.release(d)
	}
	if chans == 0 {
		return nil, errors.New("IOReport: none of the groups exist")
	}
	defer c.release(chans)
	mut := ccall(c.DictionaryCreateMutableCopy, 0, ccall(c.DictionaryGetCount, chans), chans)
	if mut == 0 {
		return nil, errors.New("IOReport: CFDictionaryCreateMutableCopy failed")
	}
	defer c.release(mut)
	subbed := new(uintptr)
	sub := ccall(l.CreateSubscription, 0, mut, uintptr(unsafe.Pointer(subbed)), 0, 0)
	runtime.KeepAlive(subbed)
	if sub == 0 || *subbed == 0 {
		return nil, errors.New("IOReport: IOReportCreateSubscription failed")
	}
	defer c.release(sub)
	defer c.release(*subbed)

	s1 := ccall(l.CreateSamples, sub, *subbed, 0)
	t1 := time.Now()
	if s1 == 0 {
		return nil, errors.New("IOReport: IOReportCreateSamples failed")
	}
	defer c.release(s1)
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-time.After(interval):
	}
	s2 := ccall(l.CreateSamples, sub, *subbed, 0)
	t2 := time.Now()
	if s2 == 0 {
		return nil, errors.New("IOReport: IOReportCreateSamples failed")
	}
	defer c.release(s2)
	delta := ccall(l.CreateSamplesDelta, s1, s2, 0)
	if delta == 0 {
		return nil, errors.New("IOReport: IOReportCreateSamplesDelta failed")
	}
	defer c.release(delta)

	rep := &DarwinIOReport{Seconds: t2.Sub(t1).Seconds()}
	key := c.cfString("IOReportChannels")
	arr := ccall(c.DictionaryGetValue, delta, key)
	c.release(key)
	if arr == 0 {
		return nil, errors.New("IOReport: samples without IOReportChannels")
	}
	n := int(ccall(c.ArrayGetCount, arr))
	for i := 0; i < n; i++ {
		ch := ccall(c.ArrayGetValueAtIndex, arr, uintptr(i))
		e := DarwinIOReportChannel{
			Group:    c.goString(ccall(l.ChannelGetGroup, ch)),
			SubGroup: c.goString(ccall(l.ChannelGetSubGroup, ch)),
			Name:     c.goString(ccall(l.ChannelGetName, ch)),
			Unit:     c.goString(ccall(l.ChannelGetUnitLabel, ch)),
		}
		format := int32(ccall(l.ChannelGetFormat, ch))
		if format == iorFormatSimple && isSoCChannel(e.Group, e.SubGroup, e.Name) {
			rep.addSoC(e.Group, e.Name, e.Unit, int64(ccall(l.SimpleGetIntegerValue, ch, 0, 0)))
		}
		if !isANEChannel(e.Group, e.SubGroup, e.Name) {
			continue
		}
		switch format {
		case iorFormatSimple:
			e.Format = "simple"
			v := int64(ccall(l.SimpleGetIntegerValue, ch, 0, 0))
			e.Value = &v
		case iorFormatState:
			e.Format = "state"
			k := int(int32(ccall(l.StateGetCount, ch)))
			for j := 0; j < k; j++ {
				e.States = append(e.States, DarwinIOReportState{
					Name:      c.goString(ccall(l.StateGetNameForIndex, ch, uintptr(j))),
					Residency: int64(ccall(l.StateGetResidency, ch, uintptr(j))),
				})
			}
		case iorFormatHistogram:
			e.Format = "histogram"
		case iorFormatSimpleArr:
			e.Format = "simple_array"
		default:
			e.Format = "unknown"
		}
		rep.Channels = append(rep.Channels, e)
	}
	rep.summarize()
	return rep, nil
}
