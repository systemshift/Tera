// Package semantic - Native neural network primitives for kernel models
// No external ML dependencies - pure Go implementation
package semantic

import (
	"encoding/binary"
	"fmt"
	"math"
)

// ============================================================================
// Neural Network Primitives (Pure Go)
// ============================================================================

// Matrix represents a 2D matrix for neural computations
type Matrix struct {
	Rows int
	Cols int
	Data []float32 // Use float32 for memory efficiency
}

// Vector is a 1D vector
type Vector []float32

// NewMatrix creates a matrix with given dimensions
func NewMatrix(rows, cols int) *Matrix {
	return &Matrix{
		Rows: rows,
		Cols: cols,
		Data: make([]float32, rows*cols),
	}
}

// NewVector creates a vector with given size
func NewVector(size int) Vector {
	return make(Vector, size)
}

// Get returns element at (i, j)
func (m *Matrix) Get(i, j int) float32 {
	return m.Data[i*m.Cols+j]
}

// Set sets element at (i, j)
func (m *Matrix) Set(i, j int, val float32) {
	m.Data[i*m.Cols+j] = val
}

// Dot computes matrix-vector multiplication: M × v
func (m *Matrix) Dot(v Vector) (Vector, error) {
	if m.Cols != len(v) {
		return nil, fmt.Errorf("dimension mismatch: %d vs %d", m.Cols, len(v))
	}

	result := NewVector(m.Rows)
	for i := 0; i < m.Rows; i++ {
		sum := float32(0)
		for j := 0; j < m.Cols; j++ {
			sum += m.Get(i, j) * v[j]
		}
		result[i] = sum
	}
	return result, nil
}

// Add adds two vectors element-wise
func (v Vector) Add(other Vector) Vector {
	result := NewVector(len(v))
	for i := range v {
		result[i] = v[i] + other[i]
	}
	return result
}

// Scale multiplies vector by scalar
func (v Vector) Scale(scalar float32) Vector {
	result := NewVector(len(v))
	for i := range v {
		result[i] = v[i] * scalar
	}
	return result
}

// Dot computes dot product of two vectors
func (v Vector) Dot(other Vector) float32 {
	sum := float32(0)
	for i := range v {
		sum += v[i] * other[i]
	}
	return sum
}

// Norm computes L2 norm
func (v Vector) Norm() float32 {
	return float32(math.Sqrt(float64(v.Dot(v))))
}

// ============================================================================
// Activation Functions
// ============================================================================

// ReLU applies ReLU activation: max(0, x)
func ReLU(v Vector) Vector {
	result := NewVector(len(v))
	for i, x := range v {
		if x > 0 {
			result[i] = x
		} else {
			result[i] = 0
		}
	}
	return result
}

// Sigmoid applies sigmoid activation: 1 / (1 + e^-x)
func Sigmoid(v Vector) Vector {
	result := NewVector(len(v))
	for i, x := range v {
		result[i] = float32(1.0 / (1.0 + math.Exp(-float64(x))))
	}
	return result
}

// Tanh applies tanh activation
func Tanh(v Vector) Vector {
	result := NewVector(len(v))
	for i, x := range v {
		result[i] = float32(math.Tanh(float64(x)))
	}
	return result
}

// Softmax applies softmax: e^x / sum(e^x)
func Softmax(v Vector) Vector {
	// Subtract max for numerical stability
	maxVal := v[0]
	for _, x := range v {
		if x > maxVal {
			maxVal = x
		}
	}

	expSum := float32(0)
	result := NewVector(len(v))
	for i, x := range v {
		exp := float32(math.Exp(float64(x - maxVal)))
		result[i] = exp
		expSum += exp
	}

	// Normalize
	for i := range result {
		result[i] /= expSum
	}
	return result
}

// ============================================================================
// Attention Mechanism (Native Implementation)
// ============================================================================

// AttentionLayer implements scaled dot-product attention
type AttentionLayer struct {
	QueryWeight *Matrix // d_model × d_k
	KeyWeight   *Matrix // d_model × d_k
	ValueWeight *Matrix // d_model × d_v
	Scale       float32 // sqrt(d_k)
}

// NewAttentionLayer creates an attention layer
func NewAttentionLayer(dModel, dK, dV int) *AttentionLayer {
	return &AttentionLayer{
		QueryWeight: NewMatrix(dK, dModel),  // dK rows × dModel cols
		KeyWeight:   NewMatrix(dK, dModel),  // dK rows × dModel cols
		ValueWeight: NewMatrix(dV, dModel),  // dV rows × dModel cols
		Scale:       float32(math.Sqrt(float64(dK))),
	}
}

// Forward computes attention: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
func (a *AttentionLayer) Forward(input Vector) (Vector, error) {
	// Q = input × W_Q
	query, err := a.QueryWeight.Dot(input)
	if err != nil {
		return nil, err
	}

	// K = input × W_K
	key, err := a.KeyWeight.Dot(input)
	if err != nil {
		return nil, err
	}

	// V = input × W_V
	value, err := a.ValueWeight.Dot(input)
	if err != nil {
		return nil, err
	}

	// Attention score = Q · K / sqrt(d_k)
	score := query.Dot(key) / a.Scale

	// Apply softmax (simplified for single-head, single-token)
	attention := float32(1.0 / (1.0 + math.Exp(-float64(score))))

	// Output = attention × V
	return value.Scale(attention), nil
}

// ============================================================================
// Multi-Layer Perceptron (MLP)
// ============================================================================

// MLPLayer represents a single dense layer
type MLPLayer struct {
	Weight     *Matrix
	Bias       Vector
	Activation string // "relu", "sigmoid", "tanh", "none"
}

// NewMLPLayer creates an MLP layer
func NewMLPLayer(inputDim, outputDim int, activation string) *MLPLayer {
	return &MLPLayer{
		Weight:     NewMatrix(outputDim, inputDim),
		Bias:       NewVector(outputDim),
		Activation: activation,
	}
}

// Forward computes: activation(W × input + b)
func (l *MLPLayer) Forward(input Vector) (Vector, error) {
	// Linear transformation
	output, err := l.Weight.Dot(input)
	if err != nil {
		return nil, err
	}
	output = output.Add(l.Bias)

	// Apply activation
	switch l.Activation {
	case "relu":
		return ReLU(output), nil
	case "sigmoid":
		return Sigmoid(output), nil
	case "tanh":
		return Tanh(output), nil
	case "none":
		return output, nil
	default:
		return output, nil
	}
}

// ============================================================================
// Complete Neural Kernel Network
// ============================================================================

// NeuralKernel is a small neural network for similarity computation
// Architecture:
//   Input (features) → Attention → MLP(128) → MLP(64) → MLP(1) → Similarity
type NeuralKernel struct {
	// Network layers
	FeatureProjection *MLPLayer      // Project features to fixed size
	Attention         *AttentionLayer // Attention over features
	Hidden1           *MLPLayer      // First hidden layer
	Hidden2           *MLPLayer      // Second hidden layer
	Output            *MLPLayer      // Output layer (1 neuron)

	// Metadata
	InputDim  int
	HiddenDim int
}

// NewNeuralKernel creates a neural kernel with given architecture
func NewNeuralKernel(inputDim, hiddenDim int) *NeuralKernel {
	// Use hiddenDim for all attention dimensions to keep things simple
	return &NeuralKernel{
		FeatureProjection: NewMLPLayer(inputDim, hiddenDim, "relu"),
		Attention:         NewAttentionLayer(hiddenDim, hiddenDim/4, hiddenDim), // Q/K smaller, V same as hidden
		Hidden1:           NewMLPLayer(hiddenDim, hiddenDim, "relu"),
		Hidden2:           NewMLPLayer(hiddenDim, hiddenDim/2, "relu"),
		Output:            NewMLPLayer(hiddenDim/2, 1, "sigmoid"),
		InputDim:          inputDim,
		HiddenDim:         hiddenDim,
	}
}

// Forward computes similarity score through the network
func (n *NeuralKernel) Forward(featuresA, featuresB Vector) (float32, error) {
	// 1. Project both feature vectors
	projA, err := n.FeatureProjection.Forward(featuresA)
	if err != nil {
		return 0, err
	}
	projB, err := n.FeatureProjection.Forward(featuresB)
	if err != nil {
		return 0, err
	}

	// 2. Compute simple interaction (element-wise product)
	// This captures relationships between features
	interaction := NewVector(len(projA))
	for i := range projA {
		interaction[i] = projA[i] * projB[i]
	}

	// 3. Apply attention to interaction
	attended, err := n.Attention.Forward(interaction)
	if err != nil {
		return 0, err
	}

	// 4. Pass through MLP layers
	h1, err := n.Hidden1.Forward(attended)
	if err != nil {
		return 0, err
	}

	h2, err := n.Hidden2.Forward(h1)
	if err != nil {
		return 0, err
	}

	// 5. Final output (similarity score)
	output, err := n.Output.Forward(h2)
	if err != nil {
		return 0, err
	}

	return output[0], nil
}

// computeInteraction creates interaction vector from two feature vectors
func (n *NeuralKernel) computeInteraction(a, b Vector) Vector {
	// Interaction = [a ⊙ b, a - b, a + b] (element-wise operations)
	// This captures: similarity, difference, and average
	size := len(a)
	result := NewVector(size * 3)

	for i := 0; i < size; i++ {
		result[i] = a[i] * b[i]           // Element-wise product
		result[size+i] = a[i] - b[i]      // Difference
		result[2*size+i] = a[i] + b[i]    // Sum
	}

	return result
}

// ============================================================================
// Quantization (8-bit weights for compact storage)
// ============================================================================

// QuantizedMatrix stores weights as int8 with scale factor
type QuantizedMatrix struct {
	Rows  int
	Cols  int
	Data  []int8   // Quantized weights
	Scale float32  // Scale factor for dequantization
	Zero  int8     // Zero point
}

// Quantize converts float32 matrix to int8
func (m *Matrix) Quantize() *QuantizedMatrix {
	// Find min/max
	minVal, maxVal := m.Data[0], m.Data[0]
	for _, val := range m.Data {
		if val < minVal {
			minVal = val
		}
		if val > maxVal {
			maxVal = val
		}
	}

	// Compute scale and zero point
	scale := (maxVal - minVal) / 255.0
	zero := int8(-minVal / scale)

	// Quantize
	qData := make([]int8, len(m.Data))
	for i, val := range m.Data {
		qData[i] = int8(val/scale) + zero
	}

	return &QuantizedMatrix{
		Rows:  m.Rows,
		Cols:  m.Cols,
		Data:  qData,
		Scale: scale,
		Zero:  zero,
	}
}

// Dequantize converts back to float32
func (q *QuantizedMatrix) Dequantize() *Matrix {
	m := NewMatrix(q.Rows, q.Cols)
	for i, qVal := range q.Data {
		m.Data[i] = float32(qVal-q.Zero) * q.Scale
	}
	return m
}

// Size returns storage size in bytes
func (q *QuantizedMatrix) Size() int {
	return len(q.Data) + 4 + 1 // int8 array + float32 scale + int8 zero
}

// ============================================================================
// Serialization (for storing in IPLD)
// ============================================================================

// SerializeMatrix converts matrix to bytes
func SerializeMatrix(m *Matrix) []byte {
	// Format: [rows:4][cols:4][data:rows*cols*4]
	buf := make([]byte, 8+len(m.Data)*4)
	binary.LittleEndian.PutUint32(buf[0:4], uint32(m.Rows))
	binary.LittleEndian.PutUint32(buf[4:8], uint32(m.Cols))

	for i, val := range m.Data {
		binary.LittleEndian.PutUint32(buf[8+i*4:], math.Float32bits(val))
	}

	return buf
}

// DeserializeMatrix converts bytes back to matrix
func DeserializeMatrix(data []byte) (*Matrix, error) {
	if len(data) < 8 {
		return nil, fmt.Errorf("invalid data: too short")
	}

	rows := int(binary.LittleEndian.Uint32(data[0:4]))
	cols := int(binary.LittleEndian.Uint32(data[4:8]))

	m := NewMatrix(rows, cols)
	for i := range m.Data {
		bits := binary.LittleEndian.Uint32(data[8+i*4:])
		m.Data[i] = math.Float32frombits(bits)
	}

	return m, nil
}

// SerializeQuantizedMatrix converts quantized matrix to bytes (4x smaller!)
func SerializeQuantizedMatrix(q *QuantizedMatrix) []byte {
	// Format: [rows:4][cols:4][scale:4][zero:1][data:rows*cols*1]
	buf := make([]byte, 13+len(q.Data))
	binary.LittleEndian.PutUint32(buf[0:4], uint32(q.Rows))
	binary.LittleEndian.PutUint32(buf[4:8], uint32(q.Cols))
	binary.LittleEndian.PutUint32(buf[8:12], math.Float32bits(q.Scale))
	buf[12] = byte(q.Zero)

	for i, val := range q.Data {
		buf[13+i] = byte(val)
	}

	return buf
}

// DeserializeQuantizedMatrix converts bytes back to quantized matrix
func DeserializeQuantizedMatrix(data []byte) (*QuantizedMatrix, error) {
	if len(data) < 13 {
		return nil, fmt.Errorf("invalid data: too short")
	}

	rows := int(binary.LittleEndian.Uint32(data[0:4]))
	cols := int(binary.LittleEndian.Uint32(data[4:8]))
	scale := math.Float32frombits(binary.LittleEndian.Uint32(data[8:12]))
	zero := int8(data[12])

	qData := make([]int8, rows*cols)
	for i := range qData {
		qData[i] = int8(data[13+i])
	}

	return &QuantizedMatrix{
		Rows:  rows,
		Cols:  cols,
		Data:  qData,
		Scale: scale,
		Zero:  zero,
	}, nil
}

