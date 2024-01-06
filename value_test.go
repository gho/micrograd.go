package main

import (
	"math"
	"testing"
)

func TestValue(t *testing.T) {
	x1 := newValue(2.0)
	w1 := newValue(-3.0)
	x2 := newValue(0.0)
	w2 := newValue(1.0)
	b := newValue(6.8813735870195432)

	x1w1 := x1.Mul(w1)
	x2w2 := x2.Mul(w2)
	n := x1w1.Add(x2w2).Add(b)
	o := n.Tanh()

	o.BackProp()

	if !floatEquals(x1.Grad, -1.5) {
		t.Fatalf("x1.Grad = %.4f is incorrect", x1.Grad)
	}

	if !floatEquals(x2.Grad, 0.5) {
		t.Fatalf("x2.Grad = %.4f is incorrect", x2.Grad)
	}
}

func TestReusedValue(t *testing.T) {
	a := newValue(-2.0)
	b := newValue(3.0)
	d := a.Mul(b)
	e := a.Add(b)
	f := d.Mul(e)

	f.BackProp()

	if !floatEquals(a.Grad, -3.0) {
		t.Fatalf("a.Grad = %.4f is incorrect", a.Grad)
	}

	if !floatEquals(b.Grad, -8.0) {
		t.Fatalf("b.Grad = %.4f is incorrect", b.Grad)
	}
}

func floatEquals(x, y float64) bool {
	return math.Abs(x-y) < 1e-2
}
