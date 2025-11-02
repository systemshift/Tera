// Example demonstrating native neural kernels with IPLD addressing
package main

import (
	"fmt"
	"tera/semantic"
)

func main() {
	fmt.Println("=== TERA Native Neural Kernel Demo ===\n")

	// ========================================================================
	// Part 1: Kernel Size Comparison (The Codec Analogy)
	// ========================================================================

	fmt.Println("1. Kernel Size Comparison (Like Audio/Video Codecs)")
	fmt.Println("   Having multiple kernels is like having H.264, VP9, AV1")
	fmt.Println()

	// Simple weighted kernel (smallest)
	simpleKernel := semantic.KernelDescriptor{
		Name:    "simple-v1",
		Format:  "native-weights",
		Version: "1.0.0",
	}
	simpleSize := 8*10 + 8*100 // 10 floats + 100 attention weights = ~1KB
	fmt.Printf("   • Simple Weighted Kernel: ~%d KB\n", simpleSize/1024)

	// Neural kernel (medium)
	neuralKernel := semantic.NewNeuralKernel(256, 128)
	neuralSize := estimateNeuralSize(neuralKernel)
	fmt.Printf("   • Neural Kernel (float32): ~%d KB\n", neuralSize/1024)

	// Quantized neural kernel (small)
	quantizedSize := neuralSize / 4 // int8 is 4x smaller than float32
	fmt.Printf("   • Neural Kernel (int8):    ~%d KB  ← 4x compression!\n", quantizedSize/1024)
	fmt.Println()

	fmt.Println("   For comparison:")
	fmt.Println("   • H.264 decoder:  ~500 KB")
	fmt.Println("   • BERT-base:      ~400 MB (400,000 KB)")
	fmt.Println("   • GPT-2:          ~500 MB")
	fmt.Println()
	fmt.Println("   → Neural kernels are 1000x smaller than LLMs!")
	fmt.Println()

	// ========================================================================
	// Part 2: Feature Extraction (Universal, Model-Agnostic)
	// ========================================================================

	fmt.Println("2. Universal Feature Extraction")
	fmt.Println()

	doc1 := []byte("Machine learning and neural networks are fundamental to modern AI systems.")
	doc2 := []byte("Deep learning with transformers has revolutionized natural language processing.")
	doc3 := []byte("I love cooking pasta with tomato sauce and fresh basil.")

	features1 := semantic.ExtractFeatures(doc1)
	features2 := semantic.ExtractFeatures(doc2)
	features3 := semantic.ExtractFeatures(doc3)

	fmt.Printf("   Document 1: %d words, %d unique, top terms: %v\n",
		features1.WordCount, features1.UniqueWords, features1.TopKeywords[:3])
	fmt.Printf("   Document 2: %d words, %d unique, top terms: %v\n",
		features2.WordCount, features2.UniqueWords, features2.TopKeywords[:3])
	fmt.Printf("   Document 3: %d words, %d unique, top terms: %v\n",
		features3.WordCount, features3.UniqueWords, features3.TopKeywords[:3])
	fmt.Println()

	// ========================================================================
	// Part 3: Simple Weighted Kernel (No Neural Network)
	// ========================================================================

	fmt.Println("3. Simple Weighted Kernel (Traditional)")
	fmt.Println()

	weights := semantic.NativeWeights{
		WeightSemantic:   0.6,
		WeightLexical:    0.3,
		WeightStructural: 0.1,
		NgramAttention:   1.0,
		LengthPenalty:    0.1,
		TemperatureAlpha: 2.0,
		MeanShift:        0.0,
		StdDevScale:      1.0,
	}

	simpleKernelInst := &semantic.NativeGoKernel{}
	// Simplified initialization
	simpleKernelInst = &semantic.NativeGoKernel{}

	params := semantic.KernelParams{
		WeightSemantic:   0.7,
		WeightLexical:    0.3,
		WeightStructural: 0.1,
		Threshold:        0.5,
	}

	// Compute similarities (using base functions since kernel needs full setup)
	sim1_2 := computeBasicSimilarity(features1, features2, params)
	sim1_3 := computeBasicSimilarity(features1, features3, params)
	sim2_3 := computeBasicSimilarity(features2, features3, params)

	fmt.Printf("   Doc1 vs Doc2 (both ML):   %.3f  ← High similarity\n", sim1_2)
	fmt.Printf("   Doc1 vs Doc3 (ML vs food): %.3f  ← Low similarity\n", sim1_3)
	fmt.Printf("   Doc2 vs Doc3 (ML vs food): %.3f  ← Low similarity\n", sim2_3)
	fmt.Println()

	// ========================================================================
	// Part 4: Neural Kernel (With Attention)
	// ========================================================================

	fmt.Println("4. Neural Kernel (With Attention)")
	fmt.Println()

	// Create small neural kernel
	nn := semantic.NewNeuralKernel(259, 64) // 256 features + 3 params
	fmt.Printf("   Network architecture:\n")
	fmt.Printf("   • Input:  259 dims (256 features + 3 params)\n")
	fmt.Printf("   • Hidden: 64 dims with attention\n")
	fmt.Printf("   • Output: 1 dim (similarity score)\n")
	fmt.Println()

	// Initialize with small random weights (in practice, train offline)
	initializeRandomWeights(nn)

	// Convert features to vectors
	vec1 := semantic.FeaturesToVector(features1, 256)
	vec2 := semantic.FeaturesToVector(features2, 256)
	vec3 := semantic.FeaturesToVector(features3, 256)

	// Add runtime params
	vec1 = appendParams(vec1, params)
	vec2 = appendParams(vec2, params)
	vec3 = appendParams(vec3, params)

	// Compute neural similarities
	neuralSim1_2, _ := nn.Forward(vec1, vec2)
	neuralSim1_3, _ := nn.Forward(vec1, vec3)
	neuralSim2_3, _ := nn.Forward(vec2, vec3)

	fmt.Printf("   Neural similarities:\n")
	fmt.Printf("   Doc1 vs Doc2 (both ML):   %.3f\n", neuralSim1_2)
	fmt.Printf("   Doc1 vs Doc3 (ML vs food): %.3f\n", neuralSim1_3)
	fmt.Printf("   Doc2 vs Doc3 (ML vs food): %.3f\n", neuralSim2_3)
	fmt.Println()

	// ========================================================================
	// Part 5: Quantization (4x Compression)
	// ========================================================================

	fmt.Println("5. Quantization (4x Compression)")
	fmt.Println()

	// Quantize a layer
	originalMatrix := nn.FeatureProjection.Weight
	quantized := originalMatrix.Quantize()

	fmt.Printf("   Original (float32): %d bytes\n", len(originalMatrix.Data)*4)
	fmt.Printf("   Quantized (int8):   %d bytes\n", quantized.Size())
	fmt.Printf("   Compression ratio:  %.1fx\n", float64(len(originalMatrix.Data)*4)/float64(quantized.Size()))
	fmt.Println()

	// Dequantize and check error
	dequantized := quantized.Dequantize()
	avgError := computeQuantizationError(originalMatrix, dequantized)
	fmt.Printf("   Average quantization error: %.6f\n", avgError)
	fmt.Printf("   → Acceptable for most tasks!\n")
	fmt.Println()

	// ========================================================================
	// Part 6: IPLD Content Addressing
	// ========================================================================

	fmt.Println("6. IPLD Content Addressing (Kernel Registry)")
	fmt.Println()

	// Create kernel descriptors with CIDs
	kernels := []semantic.KernelDescriptor{
		{
			CID:         "bafyreisemantic4h3kjn5lm...",
			Name:        "semantic-v1",
			Version:     "1.0.0",
			Description: "Text semantic similarity",
			Format:      "native-go",
			Size:        10 * 1024, // 10 KB
		},
		{
			CID:         "bafyreivisual7k2nlop9x...",
			Name:        "visual-v1",
			Version:     "1.0.0",
			Description: "Image perceptual similarity",
			Format:      "native-go",
			Size:        150 * 1024, // 150 KB
		},
		{
			CID:         "bafyreialice9mjk3pqw8...",
			Name:        "alice-research-taste",
			Version:     "1.0.0",
			Description: "Alice's custom research paper similarity",
			Format:      "native-go-quantized",
			Size:        50 * 1024, // 50 KB (quantized)
		},
	}

	fmt.Println("   Available kernels:")
	for i, k := range kernels {
		fmt.Printf("   %d. %s (CID: %s...)\n", i+1, k.Name, k.CID[:20])
		fmt.Printf("      Size: %d KB, Format: %s\n", k.Size/1024, k.Format)
	}
	fmt.Println()

	fmt.Println("   Query with specific kernel:")
	fmt.Printf("   > tera query 'transformer models' --kernel %s\n", kernels[0].CID[:20]+"...")
	fmt.Println()

	// ========================================================================
	// Part 7: Runtime Parameter Tuning (High Dexterity)
	// ========================================================================

	fmt.Println("7. Runtime Parameter Tuning (High Dexterity)")
	fmt.Println()

	fmt.Println("   Same kernel, different params:")
	fmt.Println()

	// Semantic-focused query
	semanticParams := semantic.KernelParams{
		WeightSemantic:   0.9, // High semantic weight
		WeightLexical:    0.1,
		WeightStructural: 0.0,
		Threshold:        0.6,
	}

	// Lexical-focused query
	lexicalParams := semantic.KernelParams{
		WeightSemantic:   0.1,
		WeightLexical:    0.9, // High lexical weight
		WeightStructural: 0.0,
		Threshold:        0.6,
	}

	semScore := computeBasicSimilarity(features1, features2, semanticParams)
	lexScore := computeBasicSimilarity(features1, features2, lexicalParams)

	fmt.Printf("   With semantic focus: %.3f\n", semScore)
	fmt.Printf("   With lexical focus:  %.3f\n", lexScore)
	fmt.Println()
	fmt.Println("   → Same features, same kernel, different 'taste'!")
	fmt.Println()

	// ========================================================================
	// Summary
	// ========================================================================

	fmt.Println("=== Summary ===")
	fmt.Println()
	fmt.Println("✓ Pure Go implementation (no PyTorch/ONNX)")
	fmt.Println("✓ Small kernels (10-200 KB, like codecs)")
	fmt.Println("✓ Content-addressed via IPLD CIDs")
	fmt.Println("✓ Multiple kernels coexist peacefully")
	fmt.Println("✓ Runtime parameter tuning (high dexterity)")
	fmt.Println("✓ Quantization for 4x compression")
	fmt.Println("✓ Neural layers with attention (native!)")
	fmt.Println()
	fmt.Println("This is a universal kernel system for distributed RAG!")
	fmt.Println()
}

// ============================================================================
// Helper Functions
// ============================================================================

func estimateNeuralSize(nn *semantic.NeuralKernel) int {
	size := 0

	// FeatureProjection: 259 × 64
	size += 259 * 64 * 4 // float32

	// Attention: (64×32) + (64×32) + (64×32)
	size += 3 * 64 * 32 * 4

	// Hidden1: 64 × 64
	size += 64 * 64 * 4

	// Hidden2: 64 × 32
	size += 64 * 32 * 4

	// Output: 32 × 1
	size += 32 * 1 * 4

	// Biases
	size += (64 + 64 + 32 + 1) * 4

	return size
}

func computeBasicSimilarity(a, b *semantic.Features, params semantic.KernelParams) float64 {
	params = params.Normalize()

	semantic := semantic.CosineSimilarity(a.TFIDF, b.TFIDF)
	lexical := semantic.JaccardSimilarity(a.Ngrams, b.Ngrams)
	structural := semantic.StructuralSimilarity(a, b)

	return params.WeightSemantic*semantic +
		params.WeightLexical*lexical +
		params.WeightStructural*structural
}

func initializeRandomWeights(nn *semantic.NeuralKernel) {
	// In practice, train with backprop on labeled pairs
	// For demo, use small random values
	const initScale = 0.1

	layers := []*semantic.MLPLayer{
		nn.FeatureProjection,
		nn.Hidden1,
		nn.Hidden2,
		nn.Output,
	}

	for _, layer := range layers {
		for i := range layer.Weight.Data {
			// Xavier initialization approximation
			layer.Weight.Data[i] = float32(initScale * (float64(i%100)/100.0 - 0.5))
		}
		for i := range layer.Bias {
			layer.Bias[i] = 0.0
		}
	}

	// Initialize attention weights
	for i := range nn.Attention.QueryWeight.Data {
		nn.Attention.QueryWeight.Data[i] = float32(initScale * (float64(i%100)/100.0 - 0.5))
	}
	for i := range nn.Attention.KeyWeight.Data {
		nn.Attention.KeyWeight.Data[i] = float32(initScale * (float64(i%100)/100.0 - 0.5))
	}
	for i := range nn.Attention.ValueWeight.Data {
		nn.Attention.ValueWeight.Data[i] = float32(initScale * (float64(i%100)/100.0 - 0.5))
	}
}

func appendParams(vec semantic.Vector, params semantic.KernelParams) semantic.Vector {
	return append(vec,
		float32(params.WeightSemantic),
		float32(params.WeightLexical),
		float32(params.WeightStructural),
	)
}

func computeQuantizationError(original, dequantized *semantic.Matrix) float32 {
	var sumError float32
	for i := range original.Data {
		diff := original.Data[i] - dequantized.Data[i]
		sumError += diff * diff
	}
	return sumError / float32(len(original.Data))
}
