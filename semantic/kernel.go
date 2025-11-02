package semantic

import (
	"fmt"
)

// KernelParams defines user-configurable runtime parameters for similarity computation.
// These are lightweight and can be adjusted at query time without recomputing features.
//
// Different queries can use different parameters to express different notions of "similarity".
// The neural kernel uses these params to modulate its learned weights.
type KernelParams struct {
	// Focus weights (modulate neural network behavior)
	// These control what aspects the kernel should emphasize
	WeightSemantic   float64 // Semantic meaning weight
	WeightLexical    float64 // Exact term matching weight
	WeightStructural float64 // Structural similarity weight

	// Threshold: minimum similarity to consider a match
	Threshold float64

	// Additional tuning parameters (passed to neural kernel)
	Temperature float64            // Controls score sharpness (default: 1.0)
	CustomParam map[string]float64 // Extensible params for custom kernels
}

// DefaultParams returns reasonable default parameters.
func DefaultParams() KernelParams {
	return KernelParams{
		WeightSemantic:   0.6,
		WeightLexical:    0.3,
		WeightStructural: 0.1,
		Threshold:        0.5,
		Temperature:      1.0,
		CustomParam:      make(map[string]float64),
	}
}

// SemanticFocusedParams returns parameters optimized for semantic search.
func SemanticFocusedParams() KernelParams {
	return KernelParams{
		WeightSemantic:   0.8,
		WeightLexical:    0.15,
		WeightStructural: 0.05,
		Threshold:        0.6,
		Temperature:      1.0,
		CustomParam:      make(map[string]float64),
	}
}

// LexicalFocusedParams returns parameters optimized for exact matching.
func LexicalFocusedParams() KernelParams {
	return KernelParams{
		WeightSemantic:   0.2,
		WeightLexical:    0.7,
		WeightStructural: 0.1,
		Threshold:        0.5,
		Temperature:      1.0,
		CustomParam:      make(map[string]float64),
	}
}

// Validate checks that parameters are valid.
func (p KernelParams) Validate() error {
	if p.WeightSemantic < 0 || p.WeightLexical < 0 || p.WeightStructural < 0 {
		return fmt.Errorf("weights must be non-negative")
	}

	total := p.WeightSemantic + p.WeightLexical + p.WeightStructural
	if total == 0 {
		return fmt.Errorf("at least one weight must be positive")
	}

	if p.Threshold < 0 || p.Threshold > 1 {
		return fmt.Errorf("threshold must be in [0, 1]")
	}

	if p.Temperature <= 0 {
		return fmt.Errorf("temperature must be positive")
	}

	return nil
}

// Normalize ensures weights sum to 1.
func (p KernelParams) Normalize() KernelParams {
	total := p.WeightSemantic + p.WeightLexical + p.WeightStructural

	if total == 0 {
		// Return default if all zero
		return DefaultParams()
	}

	return KernelParams{
		WeightSemantic:   p.WeightSemantic / total,
		WeightLexical:    p.WeightLexical / total,
		WeightStructural: p.WeightStructural / total,
		Threshold:        p.Threshold,
		Temperature:      p.Temperature,
		CustomParam:      p.CustomParam,
	}
}

// ToVector converts params to a vector for neural network input
func (p KernelParams) ToVector() Vector {
	vec := NewVector(8)
	vec[0] = float32(p.WeightSemantic)
	vec[1] = float32(p.WeightLexical)
	vec[2] = float32(p.WeightStructural)
	vec[3] = float32(p.Threshold)
	vec[4] = float32(p.Temperature)
	// vec[5-7] reserved for future params
	return vec
}
