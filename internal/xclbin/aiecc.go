package xclbin

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
)

// The JSON files aiecc (mlir-aie) hands to xclbinutil: kernels_*.json for --add-kernel and
// partition_*.json for --add-replace-section AIE_PARTITION:JSON. Only the fields the amdxdna
// driver path needs are read; MEM_TOPOLOGY is not used by the NPU.

type aieccKernels struct {
	PSKernels struct {
		Kernels []struct {
			Name      string `json:"name"`
			Type      string `json:"type"`
			Arguments []struct {
				Name             string `json:"name"`
				AddressQualifier string `json:"address-qualifier"`
				Offset           string `json:"offset"`
				Type             string `json:"type"`
			} `json:"arguments"`
			ExtendedData struct {
				KernelID   string `json:"dpu_kernel_id"`
				Functional string `json:"functional"`
			} `json:"extended-data"`
			Instances []struct {
				Name string `json:"name"`
			} `json:"instances"`
		} `json:"kernels"`
	} `json:"ps-kernels"`
}

type aieccPartition struct {
	AIEPartition struct {
		OpsPerCycle string `json:"operations_per_cycle"`
		Partition   struct {
			ColumnWidth  int      `json:"column_width"`
			StartColumns []uint16 `json:"start_columns"`
		} `json:"partition"`
		PDIs []struct {
			UUID      string `json:"uuid"`
			FileName  string `json:"file_name"`
			CDOGroups []struct {
				Name         string   `json:"name"`
				Type         string   `json:"type"`
				PDIID        string   `json:"pdi_id"`
				DPUKernelIDs []string `json:"dpu_kernel_ids"`
			} `json:"cdo_groups"`
		} `json:"PDIs"`
	} `json:"aie_partition"`
}

// CDO group types (enum CDO_Type).
var cdoTypes = map[string]uint8{"PRIMARY": 1, "LITE": 2, "PRE_POST": 3}

// argSize is the register map size of a kernel argument type as xclbinutil lays it out:
// pointers and 64-bit scalars take 8 bytes, everything else 4.
func argSize(typ string) int {
	if strings.Contains(typ, "*") || strings.Contains(typ, "64") {
		return 8
	}
	return 4
}

func parseNum(s string) (uint64, error) { return strconv.ParseUint(strings.TrimSpace(s), 0, 64) }

// SpecFromAiecc builds a Spec from the kernel and partition JSON aiecc writes. readPDI loads
// the PDI file a partition entry names.
func SpecFromAiecc(kernelsJSON, partitionJSON []byte, readPDI func(path string) ([]byte, error)) (Spec, error) {
	var ks aieccKernels
	if err := json.Unmarshal(kernelsJSON, &ks); err != nil {
		return Spec{}, fmt.Errorf("xclbin: kernels JSON: %w", err)
	}
	var part aieccPartition
	if err := json.Unmarshal(partitionJSON, &part); err != nil {
		return Spec{}, fmt.Errorf("xclbin: partition JSON: %w", err)
	}
	var s Spec
	for _, jk := range ks.PSKernels.Kernels {
		k := Kernel{Name: jk.Name, Type: jk.Type}
		id, err := parseNum(jk.ExtendedData.KernelID)
		if err != nil {
			return Spec{}, fmt.Errorf("xclbin: kernel %s: dpu_kernel_id %q", jk.Name, jk.ExtendedData.KernelID)
		}
		fn, err := parseNum(jk.ExtendedData.Functional)
		if err != nil {
			return Spec{}, fmt.Errorf("xclbin: kernel %s: functional %q", jk.Name, jk.ExtendedData.Functional)
		}
		k.KernelID, k.Functional = id, uint32(fn)
		for i, ja := range jk.Arguments {
			off, err := parseNum(ja.Offset)
			if err != nil {
				return Spec{}, fmt.Errorf("xclbin: kernel %s: argument %s: offset %q", jk.Name, ja.Name, ja.Offset)
			}
			a := Arg{Index: i, Name: ja.Name, Type: ja.Type, Offset: int(off), Size: argSize(ja.Type)}
			if ja.AddressQualifier == "GLOBAL" {
				a.AddressQualifier = 1
			}
			k.Args = append(k.Args, a)
		}
		s.Kernels = append(s.Kernels, k)
		for _, inst := range jk.Instances {
			s.CUs = append(s.CUs, CU{Name: jk.Name + ":" + inst.Name, Kernel: jk.Name, Type: IPPSKernel, Functional: uint32(fn), KernelID: uint32(id)})
		}
	}
	p := part.AIEPartition
	ops, err := parseNum(p.OpsPerCycle)
	if err != nil {
		return Spec{}, fmt.Errorf("xclbin: operations_per_cycle %q", p.OpsPerCycle)
	}
	s.OpsPerCycle = uint32(ops)
	s.Columns = p.Partition.ColumnWidth
	s.StartCols = p.Partition.StartColumns
	h := sha256.New()
	for _, jp := range p.PDIs {
		img, err := readPDI(jp.FileName)
		if err != nil {
			return Spec{}, err
		}
		h.Write(img)
		pdi := PDI{Image: img}
		u, err := hex.DecodeString(strings.ReplaceAll(jp.UUID, "-", ""))
		if err != nil || len(u) != 16 {
			return Spec{}, fmt.Errorf("xclbin: PDI uuid %q", jp.UUID)
		}
		copy(pdi.UUID[:], u)
		for _, jg := range jp.CDOGroups {
			g := CDOGroup{Name: jg.Name, Type: cdoTypes[jg.Type]}
			if g.PDIID, err = parseNum(jg.PDIID); err != nil {
				return Spec{}, fmt.Errorf("xclbin: pdi_id %q", jg.PDIID)
			}
			for _, sid := range jg.DPUKernelIDs {
				id, err := parseNum(sid)
				if err != nil {
					return Spec{}, fmt.Errorf("xclbin: dpu_kernel_id %q", sid)
				}
				g.KernelIDs = append(g.KernelIDs, id)
			}
			pdi.Groups = append(pdi.Groups, g)
		}
		s.PDIs = append(s.PDIs, pdi)
	}
	// The container UUID only has to be unique per design: derive it from the PDIs.
	copy(s.UUID[:], h.Sum(nil))
	return s, nil
}
