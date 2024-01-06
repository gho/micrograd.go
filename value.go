package main

import (
	"fmt"
	"math"
	"slices"
)

type (
	set map[*Value]struct{}

	Value struct {
		Data       float64
		Grad       float64
		Parents    set
		backPropFn func()
	}
)

func (v *Value) String() string {
	return fmt.Sprintf("Value{Data:%.4f, Grad:%.4f}", v.Data, v.Grad)
}

func newValue(data float64, parents ...*Value) *Value {
	m := make(set)
	for _, parent := range parents {
		m[parent] = struct{}{}
	}
	return &Value{
		Data:       data,
		Parents:    m,
		backPropFn: func() {},
	}
}

func (v *Value) Add(other *Value) *Value {
	sum := newValue(v.Data+other.Data, v, other)
	sum.backPropFn = func() {
		v.Grad += sum.Grad
		other.Grad += sum.Grad
	}
	return sum
}

func (v *Value) Sub(other *Value) *Value {
	return v.Add(other.Negate())
}

func (v *Value) Mul(other *Value) *Value {
	prod := newValue(v.Data*other.Data, v, other)
	prod.backPropFn = func() {
		v.Grad += prod.Grad * other.Data
		other.Grad += prod.Grad * v.Data
	}
	return prod
}

func (v *Value) Negate() *Value {
	return v.Mul(newValue(-1.0))
}

func (v *Value) Tanh() *Value {
	tanh := newValue(math.Tanh(v.Data), v)
	tanh.backPropFn = func() {
		v.Grad += (1 - math.Pow(tanh.Data, 2)) * tanh.Grad
	}
	return tanh
}

func (v *Value) Pow(n float64) *Value {
	res := newValue(math.Pow(v.Data, n), v)
	res.backPropFn = func() {
		v.Grad += n * math.Pow(v.Data, n-1) * res.Grad
	}
	return res
}

func (v *Value) BackProp() {
	var visit func(v *Value)

	var ordered []*Value
	visited := make(set)
	visit = func(v *Value) {
		if _, ok := visited[v]; !ok {
			visited[v] = struct{}{}
			for parent, _ := range v.Parents {
				visit(parent)
			}
			ordered = append(ordered, v)
		}
	}

	v.Grad = 1.0
	visit(v)
	slices.Reverse(ordered)

	for _, v := range ordered {
		v.backPropFn()
	}
}
