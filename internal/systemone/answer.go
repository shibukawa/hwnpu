package systemone

import (
	"bytes"
	"encoding/json"
	"math"
	"strconv"

	"github.com/shibukawa/hwnpu/internal/pyjson"
)

// Answer is the answer to one question, written as llama.cpp writes it.
type Answer struct {
	Type          string
	Choice        string    // choice: the most likely option
	Score         float64   // score: the median level
	Noul          float64   // noul: the probability of true
	Options       []Option  // the options, for probabilities and legend
	Probabilities []float64 // by option
	Confidence    float64   // choice and score: the probability of the answer
}

// Decide turns the logits of the options' labels (the best of each group) into an answer:
// a softmax at temperature, then for a choice the most likely option, for a score the median
// level (the first whose cumulative probability reaches one half) and for a noul the
// probability of true; the confidence is the answer's probability, as the d1 api.py has it.
func Decide(q Question, logits []float64, temperature float64) Answer {
	mx := math.Inf(-1)
	for _, l := range logits {
		mx = math.Max(mx, l)
	}
	p := make([]float64, len(logits))
	var sum float64
	for i, l := range logits {
		p[i] = math.Exp((l - mx) / temperature)
		sum += p[i]
	}
	best := 0
	for i := range p {
		p[i] /= sum
		if p[i] > p[best] {
			best = i
		}
	}
	a := Answer{Type: q.Type, Options: q.Options, Probabilities: p, Confidence: p[best]}
	switch q.Type {
	case "choice":
		a.Choice = q.Options[best].Key
	case "score":
		mass, level := 0.0, len(p)-1
		for i := 0; i < len(p) && mass < 0.5; i++ {
			mass += p[i]
			level = i
		}
		a.Score = float64(level)
	case "noul":
		for i, o := range q.Options {
			if o.Key == "true" {
				a.Noul = p[i]
			}
		}
	}
	return a
}

// MarshalJSON writes the answer with its keys in llama.cpp's order and the options in theirs.
func (a Answer) MarshalJSON() ([]byte, error) {
	var b bytes.Buffer
	b.WriteString(`{"type":` + strconv.Quote(a.Type))
	num := func(key string, v float64) {
		b.WriteString(`,"` + key + `":` + strconv.FormatFloat(v, 'g', -1, 64))
	}
	object := func(key string, value func(i int)) {
		b.WriteString(`,"` + key + `":{`)
		for i, o := range a.Options {
			if i > 0 {
				b.WriteByte(',')
			}
			k, _ := json.Marshal(o.Key)
			b.Write(k)
			b.WriteByte(':')
			value(i)
		}
		b.WriteByte('}')
	}
	probs := func(i int) { b.WriteString(strconv.FormatFloat(a.Probabilities[i], 'g', -1, 64)) }
	switch a.Type {
	case "noul":
		num("noul", a.Noul)
	case "choice":
		k, _ := json.Marshal(a.Choice)
		b.WriteString(`,"choice":` + string(k))
		object("probabilities", probs)
		num("confidence", a.Confidence)
	default:
		num("score", a.Score)
		object("legend", func(i int) {
			d := a.Options[i].Description
			if d == nil {
				d = &pyjson.Value{Kind: '0'}
			}
			b.WriteString(d.String())
		})
		object("probabilities", probs)
		num("confidence", a.Confidence)
	}
	b.WriteByte('}')
	return b.Bytes(), nil
}
