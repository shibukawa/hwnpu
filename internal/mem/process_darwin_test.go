package mem

import "testing"

func TestProcessMemory(t *testing.T) {
	m, ok := ProcessMemory()
	if !ok {
		t.Fatal("ProcessMemory failed on darwin")
	}
	if m.PhysFootprint <= 0 {
		t.Errorf("PhysFootprint = %d, want > 0", m.PhysFootprint)
	}
	if m.PeakFootprint < m.PhysFootprint {
		t.Errorf("PeakFootprint = %d < PhysFootprint = %d", m.PeakFootprint, m.PhysFootprint)
	}
	t.Logf("phys=%.1f MiB peak=%.1f MiB ext=%.1f MiB neural_fp=%d neural_nofp=%d neural_peak=%d",
		float64(m.PhysFootprint)/(1<<20), float64(m.PeakFootprint)/(1<<20),
		float64(m.External)/(1<<20), m.NeuralFootprint, m.NeuralNoFootprint, m.NeuralNoFootprintPeak)
}
