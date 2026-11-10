package chat

import "testing"

// moss-soundeffect:8g (8GB, 8gb) picks the lut8 model, :16g and the family on 16 GB the fp16
// one; the use sfx picks by the machine.
func TestPickSoundEffect(t *testing.T) {
	for _, c := range []struct {
		name    string
		machine int
		want    string
	}{
		{"moss-soundeffect:8g", 16, "MOSS-SoundEffect-v2.0-lut8"},
		{"moss-soundeffect:8GB", 16, "MOSS-SoundEffect-v2.0-lut8"},
		{"MOSS-SoundEffect:8gb", 16, "MOSS-SoundEffect-v2.0-lut8"},
		{"moss-soundeffect:16g", 8, "MOSS-SoundEffect-v2.0"},
		{"moss-soundeffect", 16, "MOSS-SoundEffect-v2.0"},
		{"moss-soundeffect", 8, "MOSS-SoundEffect-v2.0-lut8"},
		{"sfx", 8, "MOSS-SoundEffect-v2.0-lut8"},
		{"sound-effects", 32, "MOSS-SoundEffect-v2.0"},
	} {
		m, _, ok, err := Pick(c.name, c.machine)
		if !ok || err != nil || m.Name != c.want {
			t.Errorf("%s on %d GB: %s (%v, %v), want %s", c.name, c.machine, m.Name, ok, err, c.want)
			continue
		}
		if m.Kind != KindSFX || m.Repo != "OpenMOSS-Team/MOSS-SoundEffect-v2.0" || m.DirName() != "MOSS-SoundEffect-v2.0" {
			t.Errorf("%s: kind %s, repo %s", m.Name, m.Kind, m.Repo)
		}
	}
}
