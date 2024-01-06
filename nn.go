package main

import (
	"log"
	"math/rand"
)

type neuron struct {
	w []*Value
	b *Value
}

func newNeuron(nin int) *neuron {
	return &neuron{
		w: randWeights(nin),
		b: randWeights(1)[0],
	}
}

func randWeights(nin int) []*Value {
	ws := make([]*Value, nin)
	for i := 0; i < nin; i++ {
		ws[i] = newValue(2*rand.Float64() - 1)
	}
	return ws
}

func (n *neuron) infer(x []*Value) *Value {
	if len(x) != len(n.w) {
		log.Fatalf("expected %d inputs but got %d", len(n.w), len(x))
	}
	res := newValue(0)
	for i, xi := range x {
		res = res.Add(xi.Mul(n.w[i]))
	}
	return res.Add(n.b).Tanh()
}

func (n *neuron) parameters() []*Value {
	return append(n.w, n.b)
}

type layer struct {
	neurons []*neuron
}

func newLayer(nin, nout int) *layer {
	ns := make([]*neuron, nout)
	for i := range ns {
		ns[i] = newNeuron(nin)
	}
	return &layer{neurons: ns}
}

func (l *layer) infer(x []*Value) []*Value {
	res := make([]*Value, len(l.neurons))
	for i, n := range l.neurons {
		res[i] = n.infer(x)
	}
	return res
}

func (l *layer) parameters() []*Value {
	var ps []*Value
	for _, n := range l.neurons {
		for _, p := range n.parameters() {
			ps = append(ps, p)
		}
	}
	return ps
}

type MLP struct {
	layers []*layer
}

func newMLP(nin int, nouts []int) *MLP {
	sz := append([]int{nin}, nouts...)
	ls := make([]*layer, len(nouts))
	for i := range ls {
		ls[i] = newLayer(sz[i], sz[i+1])
	}
	return &MLP{layers: ls}
}

func (mlp *MLP) parameters() []*Value {
	var ps []*Value
	for _, l := range mlp.layers {
		for _, p := range l.parameters() {
			ps = append(ps, p)
		}
	}
	return ps
}

func (mlp *MLP) Train(xs [][]*Value, ys []*Value) {
	for {
		// forward pass
		ypred := make([][]*Value, len(xs))
		for i := range xs {
			ypred[i] = mlp.Infer(xs[i])
		}

		// calculate loss
		loss := newValue(0)
		for i := range ys {
			loss = loss.Add(ypred[i][0].Sub(ys[i]).Pow(2))
		}

		// backward pass
		loss.BackProp()

		// gradient descent
		for _, p := range mlp.parameters() {
			p.Data -= 0.05 * p.Grad
			p.Grad = 0
		}

		// good enough?
		if loss.Data < 1e-5 {
			break
		}
	}
}

func (mlp *MLP) Infer(x []*Value) []*Value {
	for _, l := range mlp.layers {
		x = l.infer(x)
	}
	return x
}
