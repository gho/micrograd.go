package main

import (
	"testing"
)

func TestMLP(t *testing.T) {
	mlp := newMLP(3, []int{4, 4, 1})

	mlp.Train(
		[][]*Value{
			newValues(2.0, 3.0, -1.0),
			newValues(3.0, -1.0, 0.5),
			newValues(0.5, 1.0, 1.0),
			newValues(1.0, 1.0, -1.0),
		},
		newValues(1.0, -1.0, -1.0, 1.0),
	)

	a := mlp.Infer(newValues(0.4, 1.2, 0.9))[0]
	if !floatEquals(a.Data, -1.0) {
		t.Fatalf("inferred a = %f is incorrect", a.Data)
	}

	b := mlp.Infer(newValues(1.1, 0.9, -0.8))[0]
	if !floatEquals(b.Data, 1.0) {
		t.Fatalf("inferred b = %f is incorrect", b.Data)
	}
}

func newValues(data ...float64) []*Value {
	vs := make([]*Value, len(data))
	for i := range vs {
		vs[i] = newValue(data[i])
	}
	return vs
}
