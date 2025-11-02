// Package semantic - Neural kernel models with IPLD content addressing
package semantic

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
)

// KernelDescriptor is an IPLD-addressable kernel model descriptor.
// This allows kernels to be content-addressed and shared across the network.
type KernelDescriptor struct {
	// IPLD metadata
	CID     string `json:"/"` // Content ID (e.g., "bafyreib...")
	Version string `json:"version"`

	// Model metadata
	Name        string        `json:"name"`
	Description string        `json:"description"`
	InputSchema FeatureSchema `json:"input_schema"`

	// Model weights
	Format     string `json:"format"`      // "native-go", "native-go-quantized"
	Weights    []byte `json:"weights"`     // Serialized model weights
	WeightsCID string `json:"weights_cid"` // Or separate CID for large weights
	Size       int64  `json:"size"`

	// Provenance
	Author    string `json:"author"`
	License   string `json:"license"`
	TrainedOn string `json:"trained_on"`

	// Runtime hints
	Deterministic bool    `json:"deterministic"` // Always same output for same input
	Complexity    string  `json:"complexity"`    // "O(n)", "O(n²)", etc.
	AvgLatency    float64 `json:"avg_latency_ms"`
}

// FeatureSchema describes what features the kernel expects
type FeatureSchema struct {
	Modalities []string          `json:"modalities"` // ["text", "image", etc.]
	VectorSize int               `json:"vector_size"` // Expected input dimension
	Required   []string          `json:"required"`   // Must-have features
	Optional   []string          `json:"optional"`   // Nice-to-have features
	Metadata   map[string]string `json:"metadata"`   // Extra context
}

// KernelModel is the runtime representation of a neural kernel.
// It's loaded from a KernelDescriptor and cached locally.
type KernelModel interface {
	// ComputeSimilarity is the core function
	ComputeSimilarity(a, b *FeatureVector, params KernelParams) (float64, error)

	// Metadata
	CID() string
	Name() string
	Version() string

	// Validation
	ValidateFeatures(f *FeatureVector) error
	CompatibleWith(schema FeatureSchema) bool
}

// ============================================================================
// Native Go Neural Kernel (Pure Go, No Dependencies)
// ============================================================================

// NativeGoKernel is a kernel implemented in pure Go with neural layers.
type NativeGoKernel struct {
	descriptor    KernelDescriptor
	neuralNetwork *NeuralKernel
}

// NewNativeKernel creates a native Go kernel from a descriptor
func NewNativeKernel(desc KernelDescriptor) (*NativeGoKernel, error) {
	// Determine input size from schema
	inputSize := desc.InputSchema.VectorSize
	if inputSize == 0 {
		inputSize = 512 // Default
	}

	// Create neural network
	// Input: feature vector + params vector
	paramSize := 8 // KernelParams.ToVector() size
	networkInputSize := inputSize + paramSize

	hiddenSize := 128
	if inputSize < 256 {
		hiddenSize = 64 // Smaller network for smaller inputs
	}

	nn := NewNeuralKernel(networkInputSize, hiddenSize)

	// Load weights if provided
	if len(desc.Weights) > 0 {
		if err := loadWeights(nn, desc.Weights); err != nil {
			return nil, fmt.Errorf("failed to load weights: %w", err)
		}
	} else {
		// Initialize with defaults (Xavier init done in NewNeuralKernel)
		initializeDefaultWeights(nn)
	}

	return &NativeGoKernel{
		descriptor:    desc,
		neuralNetwork: nn,
	}, nil
}

// ComputeSimilarity implements the neural similarity computation
func (k *NativeGoKernel) ComputeSimilarity(a, b *FeatureVector, params KernelParams) (float64, error) {
	// Validate features
	if err := k.ValidateFeatures(a); err != nil {
		return 0, err
	}
	if err := k.ValidateFeatures(b); err != nil {
		return 0, err
	}

	// Normalize params
	params = params.Normalize()

	// Convert params to vector
	paramsVec := params.ToVector()

	// Concatenate feature vectors with params
	// [featureA, params] and [featureB, params]
	vecA := append(a.Data, paramsVec...)
	vecB := append(b.Data, paramsVec...)

	// Run through neural network
	score, err := k.neuralNetwork.Forward(vecA, vecB)
	if err != nil {
		return 0, fmt.Errorf("neural forward pass failed: %w", err)
	}

	return float64(score), nil
}

// CID returns the content identifier
func (k *NativeGoKernel) CID() string {
	return k.descriptor.CID
}

// Name returns kernel name
func (k *NativeGoKernel) Name() string {
	return k.descriptor.Name
}

// Version returns kernel version
func (k *NativeGoKernel) Version() string {
	return k.descriptor.Version
}

// ValidateFeatures checks if features are compatible
func (k *NativeGoKernel) ValidateFeatures(f *FeatureVector) error {
	schema := k.descriptor.InputSchema

	// Check modality if specified
	if len(schema.Modalities) > 0 {
		found := false
		for _, m := range schema.Modalities {
			if f.Modality == m {
				found = true
				break
			}
		}
		if !found {
			return fmt.Errorf("unsupported modality: %s (expected: %v)", f.Modality, schema.Modalities)
		}
	}

	// Check vector size
	if schema.VectorSize > 0 && len(f.Data) != schema.VectorSize {
		return fmt.Errorf("dimension mismatch: got %d, expected %d", len(f.Data), schema.VectorSize)
	}

	return nil
}

// CompatibleWith checks schema compatibility
func (k *NativeGoKernel) CompatibleWith(schema FeatureSchema) bool {
	mySchema := k.descriptor.InputSchema

	// Check modalities
	if len(schema.Modalities) > 0 && len(mySchema.Modalities) > 0 {
		hasCommon := false
		for _, m1 := range schema.Modalities {
			for _, m2 := range mySchema.Modalities {
				if m1 == m2 {
					hasCommon = true
					break
				}
			}
		}
		if !hasCommon {
			return false
		}
	}

	// Check vector size
	if schema.VectorSize > 0 && mySchema.VectorSize > 0 {
		if schema.VectorSize != mySchema.VectorSize {
			return false
		}
	}

	return true
}

// ============================================================================
// Kernel Registry and Caching
// ============================================================================

// KernelRegistry manages available kernel models.
// It handles downloading, caching, and instantiation.
type KernelRegistry struct {
	cache   map[string]KernelModel // CID -> Model
	storage string                 // Cache directory
}

// NewKernelRegistry creates a registry with local cache
func NewKernelRegistry(cacheDir string) *KernelRegistry {
	return &KernelRegistry{
		cache:   make(map[string]KernelModel),
		storage: cacheDir,
	}
}

// Get retrieves a kernel by CID (download if needed)
func (r *KernelRegistry) Get(cid string) (KernelModel, error) {
	// Check cache first
	if model, ok := r.cache[cid]; ok {
		return model, nil
	}

	// Load from local storage
	model, err := r.loadFromDisk(cid)
	if err == nil {
		r.cache[cid] = model
		return model, nil
	}

	// Download from IPFS/network
	model, err = r.downloadKernel(cid)
	if err != nil {
		return nil, fmt.Errorf("failed to get kernel %s: %w", cid, err)
	}

	// Cache it
	r.cache[cid] = model
	r.saveToDisk(cid, model)

	return model, nil
}

// Register adds a kernel to the registry
func (r *KernelRegistry) Register(model KernelModel) error {
	r.cache[model.CID()] = model
	return r.saveToDisk(model.CID(), model)
}

// List returns all available kernels
func (r *KernelRegistry) List() []string {
	cids := make([]string, 0, len(r.cache))
	for cid := range r.cache {
		cids = append(cids, cid)
	}
	return cids
}

// ============================================================================
// Standard Kernels (Shipped with TERA)
// ============================================================================

// StandardKernels returns the default kernel set
func StandardKernels() []KernelDescriptor {
	return []KernelDescriptor{
		{
			CID:         computeCID("semantic-text-v1", []byte{}),
			Version:     "1.0.0",
			Name:        "semantic-text-v1",
			Description: "Semantic similarity optimized for text",
			InputSchema: FeatureSchema{
				Modalities: []string{"text"},
				VectorSize: 512,
				Required:   []string{"text-features"},
			},
			Format:        "native-go",
			Deterministic: true,
			Complexity:    "O(n)",
			Size:          100 * 1024, // ~100 KB
		},
		{
			CID:         computeCID("visual-v1", []byte{}),
			Version:     "1.0.0",
			Name:        "visual-v1",
			Description: "Perceptual similarity for images",
			InputSchema: FeatureSchema{
				Modalities: []string{"image"},
				VectorSize: 512,
				Required:   []string{"image-features"},
			},
			Format:        "native-go",
			Deterministic: true,
			Complexity:    "O(n)",
			Size:          150 * 1024, // ~150 KB
		},
		{
			CID:         computeCID("code-v1", []byte{}),
			Version:     "1.0.0",
			Name:        "code-v1",
			Description: "Structural similarity for source code",
			InputSchema: FeatureSchema{
				Modalities: []string{"code"},
				VectorSize: 512,
				Required:   []string{"code-features"},
			},
			Format:        "native-go",
			Deterministic: true,
			Complexity:    "O(n)",
			Size:          120 * 1024, // ~120 KB
		},
	}
}

// ============================================================================
// Utilities
// ============================================================================

// computeCID generates a CID for a kernel (simplified)
func computeCID(name string, weights []byte) string {
	hash := sha256.Sum256(append([]byte(name), weights...))
	// In real implementation, use proper IPLD CID with multihash
	return "bafyrei" + hex.EncodeToString(hash[:])[:16]
}

func loadWeights(nn *NeuralKernel, data []byte) error {
	// TODO: Deserialize weights from binary format
	// For now, weights are initialized by default
	return nil
}

func initializeDefaultWeights(nn *NeuralKernel) {
	// Initialize with small random values (Xavier-like initialization)
	const initScale = 0.1

	layers := []*MLPLayer{
		nn.FeatureProjection,
		nn.Hidden1,
		nn.Hidden2,
		nn.Output,
	}

	for _, layer := range layers {
		fanIn := float32(layer.Weight.Cols)
		fanOut := float32(layer.Weight.Rows)
		scale := float32(initScale * 2.0 / (fanIn + fanOut))

		for i := range layer.Weight.Data {
			// Simple pseudo-random (not cryptographic)
			layer.Weight.Data[i] = float32(i%100)/100.0*scale*2 - scale
		}
		for i := range layer.Bias {
			layer.Bias[i] = 0.0
		}
	}

	// Initialize attention weights
	for i := range nn.Attention.QueryWeight.Data {
		nn.Attention.QueryWeight.Data[i] = float32(i%100)/100.0*initScale*2 - initScale
	}
	for i := range nn.Attention.KeyWeight.Data {
		nn.Attention.KeyWeight.Data[i] = float32(i%100)/100.0*initScale*2 - initScale
	}
	for i := range nn.Attention.ValueWeight.Data {
		nn.Attention.ValueWeight.Data[i] = float32(i%100)/100.0*initScale*2 - initScale
	}
}

func (r *KernelRegistry) loadFromDisk(cid string) (KernelModel, error) {
	// TODO: Load from cache directory
	return nil, fmt.Errorf("not in cache")
}

func (r *KernelRegistry) downloadKernel(cid string) (KernelModel, error) {
	// TODO: Fetch from IPFS or TERA network
	return nil, fmt.Errorf("download not implemented")
}

func (r *KernelRegistry) saveToDisk(cid string, model KernelModel) error {
	// TODO: Persist to cache directory
	return nil
}
