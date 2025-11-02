// Package semantic - Universal feature extraction for all modalities
// This replaces the old text-only features.go with a clean universal design
package semantic

import (
	"crypto/sha256"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"strings"
	"unicode"
)

// ============================================================================
// Universal Feature Vector (The ONLY representation)
// ============================================================================

// FeatureVector is the universal representation for ALL content types.
// It's a fixed-size dense vector that can be efficiently transmitted and
// compared using neural kernels.
type FeatureVector struct {
	// Metadata
	Modality string // "text", "image", "audio", "code", "binary", "unknown"
	Size     int    // Dimensionality

	// Dense feature representation (THIS IS THE ONLY DATA)
	Data Vector // Fixed-size float32 vector

	// Hash for deduplication
	Hash string // SHA-256 of original content
}

// ============================================================================
// Universal Feature Extraction (ONE entry point)
// ============================================================================

// ExtractFeatures is the ONLY function for feature extraction.
// It detects the modality and produces a universal fixed-size vector.
func ExtractFeatures(content []byte, filename string) (*FeatureVector, error) {
	// 1. Detect modality
	modality := detectModality(content, filename)

	// 2. Extract modality-specific features → universal vector
	var data Vector
	var err error

	switch modality {
	case "text":
		data, err = extractTextVector(content)
	case "image":
		data, err = extractImageVector(content)
	case "audio":
		data, err = extractAudioVector(content)
	case "code":
		data, err = extractCodeVector(content, filename)
	case "binary":
		data, err = extractBinaryVector(content)
	default:
		data, err = extractGenericVector(content)
	}

	if err != nil {
		return nil, fmt.Errorf("feature extraction failed: %w", err)
	}

	// 3. Compute hash
	hash := fmt.Sprintf("%x", sha256.Sum256(content))

	return &FeatureVector{
		Modality: modality,
		Size:     len(data),
		Data:     data,
		Hash:     hash,
	}, nil
}

// ============================================================================
// Modality Detection
// ============================================================================

func detectModality(content []byte, filename string) string {
	// Check magic bytes first
	if len(content) < 4 {
		return "unknown"
	}

	// Image formats
	if content[0] == 0xFF && content[1] == 0xD8 {
		return "image" // JPEG
	}
	if content[0] == 0x89 && content[1] == 0x50 && content[2] == 0x4E && content[3] == 0x47 {
		return "image" // PNG
	}
	if content[0] == 0x47 && content[1] == 0x49 && content[2] == 0x46 {
		return "image" // GIF
	}

	// Check filename extension
	lower := strings.ToLower(filename)
	if strings.HasSuffix(lower, ".go") || strings.HasSuffix(lower, ".py") ||
		strings.HasSuffix(lower, ".js") || strings.HasSuffix(lower, ".rs") ||
		strings.HasSuffix(lower, ".c") || strings.HasSuffix(lower, ".cpp") {
		return "code"
	}

	if strings.HasSuffix(lower, ".txt") || strings.HasSuffix(lower, ".md") {
		return "text"
	}

	if strings.HasSuffix(lower, ".wav") || strings.HasSuffix(lower, ".mp3") ||
		strings.HasSuffix(lower, ".flac") || strings.HasSuffix(lower, ".ogg") {
		return "audio"
	}

	// Heuristic: if mostly printable ASCII, it's text
	printable := 0
	for i := 0; i < len(content) && i < 1000; i++ {
		if content[i] >= 32 && content[i] <= 126 || content[i] == '\n' || content[i] == '\r' || content[i] == '\t' {
			printable++
		}
	}

	if printable > 800 { // 80% printable
		return "text"
	}

	return "binary"
}

// ============================================================================
// Text Feature Extraction → Universal Vector
// ============================================================================

const (
	TextVectorSize = 512 // Fixed size for all text
)

func extractTextVector(content []byte) (Vector, error) {
	text := string(content)
	vec := NewVector(TextVectorSize)

	// 1. Tokenize
	words := tokenize(text)
	if len(words) == 0 {
		return vec, nil
	}

	// 2. Compute term frequencies
	tf := computeTF(words)

	// 3. Hash terms into fixed-size vector (feature hashing)
	// This replaces sparse TF-IDF with dense representation
	for term, freq := range tf {
		hash := hashTerm(term) % TextVectorSize
		vec[hash] += float32(freq)
	}

	// 4. Add statistical features to last positions
	statsOffset := TextVectorSize - 16
	vec[statsOffset+0] = normalizeCount(len(words), 10000)       // Word count
	vec[statsOffset+1] = normalizeCount(len(text), 100000)       // Char count
	vec[statsOffset+2] = normalizeCount(len(tf), 5000)           // Unique words
	vec[statsOffset+3] = avgWordLength(words)                     // Avg word length
	vec[statsOffset+4] = sentenceCount(text)                      // Sentence count
	vec[statsOffset+5] = uppercaseRatio(text)                     // Uppercase ratio
	vec[statsOffset+6] = punctuationRatio(text)                   // Punctuation ratio
	vec[statsOffset+7] = digitRatio(text)                         // Digit ratio

	// 5. Add character n-gram features (like your lexical similarity)
	for i := 0; i < len(text)-2 && i < 1000; i++ {
		ngram := text[i : i+3]
		hash := hashTerm(ngram) % (TextVectorSize - 16)
		vec[hash] += 0.1 // Smaller weight for n-grams
	}

	// 6. Normalize to unit vector
	norm := vec.Norm()
	if norm > 0 {
		for i := range vec {
			vec[i] /= norm
		}
	}

	return vec, nil
}

// ============================================================================
// Image Feature Extraction → Universal Vector
// ============================================================================

const (
	ImageVectorSize = 512 // Fixed size for all images
)

func extractImageVector(content []byte) (Vector, error) {
	vec := NewVector(ImageVectorSize)

	// Decode image
	img, _, err := image.Decode(strings.NewReader(string(content)))
	if err != nil {
		return extractGenericVector(content) // Fallback to generic
	}

	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()

	// 1. Color histogram (256 positions, 4 bins per channel = 64 bins)
	colorBins := make([]float32, 64)
	totalPixels := 0

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, _ := img.At(x, y).RGBA()

			// Quantize to 4 levels per channel
			rBin := (r >> 14) & 0x3
			gBin := (g >> 14) & 0x3
			bBin := (b >> 14) & 0x3

			binIdx := rBin*16 + gBin*4 + bBin
			colorBins[binIdx]++
			totalPixels++
		}
	}

	// Normalize and copy to vector
	for i := 0; i < 64; i++ {
		vec[i] = colorBins[i] / float32(totalPixels)
	}

	// 2. Edge histogram (64-128: 64 positions)
	edgeBins := computeEdgeHistogram(img)
	copy(vec[64:128], edgeBins)

	// 3. Spatial features (128-192: 64 positions)
	vec[128] = normalizeCount(width, 4096)
	vec[129] = normalizeCount(height, 4096)
	vec[130] = float32(width) / float32(height) // Aspect ratio
	vec[131] = float32(totalPixels) / 10000000.0 // Megapixels

	// 4. Texture features (192-256: 64 positions)
	textures := computeTextureFeatures(img)
	copy(vec[192:256], textures)

	// 5. Remaining positions: reserved for future features

	return vec, nil
}

// ============================================================================
// Audio Feature Extraction → Universal Vector
// ============================================================================

const (
	AudioVectorSize = 512 // Fixed size for all audio
)

func extractAudioVector(content []byte) (Vector, error) {
	// TODO: Implement proper audio feature extraction
	// For now, use generic byte-level features
	return extractGenericVector(content)
}

// ============================================================================
// Code Feature Extraction → Universal Vector
// ============================================================================

const (
	CodeVectorSize = 512 // Fixed size for all code
)

func extractCodeVector(content []byte, filename string) (Vector, error) {
	code := string(content)
	vec := NewVector(CodeVectorSize)

	// 1. Token frequency (similar to text but code-aware)
	tokens := tokenizeCode(code)
	tf := computeTF(tokens)

	for token, freq := range tf {
		hash := hashTerm(token) % (CodeVectorSize - 32)
		vec[hash] += float32(freq)
	}

	// 2. Code-specific features (last 32 positions)
	statsOffset := CodeVectorSize - 32
	vec[statsOffset+0] = normalizeCount(len(strings.Split(code, "\n")), 10000) // Line count
	vec[statsOffset+1] = normalizeCount(countOccurrences(code, "func"), 1000)   // Function count
	vec[statsOffset+2] = normalizeCount(countOccurrences(code, "class"), 1000)  // Class count
	vec[statsOffset+3] = normalizeCount(countOccurrences(code, "import"), 100)  // Import count
	vec[statsOffset+4] = normalizeCount(countOccurrences(code, "if"), 1000)     // Conditional count
	vec[statsOffset+5] = normalizeCount(countOccurrences(code, "for"), 1000)    // Loop count
	vec[statsOffset+6] = indentationDepth(code)                                  // Max indent
	vec[statsOffset+7] = commentRatio(code)                                      // Comment ratio

	// 3. Normalize
	norm := vec.Norm()
	if norm > 0 {
		for i := range vec {
			vec[i] /= norm
		}
	}

	return vec, nil
}

// ============================================================================
// Binary/Generic Feature Extraction → Universal Vector
// ============================================================================

const (
	BinaryVectorSize = 512 // Fixed size for all binary
)

func extractBinaryVector(content []byte) (Vector, error) {
	return extractGenericVector(content)
}

func extractGenericVector(content []byte) (Vector, error) {
	vec := NewVector(BinaryVectorSize)

	// 1. Byte histogram (256 bins)
	for _, b := range content {
		vec[b] += 1.0
	}

	// Normalize by content length
	length := float32(len(content))
	if length > 0 {
		for i := 0; i < 256; i++ {
			vec[i] /= length
		}
	}

	// 2. Entropy and statistics (256-512)
	vec[256] = entropy(content)
	vec[257] = normalizeCount(len(content), 100000000) // File size
	vec[258] = float32(countZeros(content)) / length   // Zero ratio

	return vec, nil
}

// ============================================================================
// Helper Functions
// ============================================================================

func tokenize(text string) []string {
	text = strings.ToLower(text)
	words := []string{}
	current := strings.Builder{}

	for _, r := range text {
		if unicode.IsLetter(r) || unicode.IsNumber(r) {
			current.WriteRune(r)
		} else if current.Len() > 0 {
			words = append(words, current.String())
			current.Reset()
		}
	}

	if current.Len() > 0 {
		words = append(words, current.String())
	}

	return words
}

func tokenizeCode(code string) []string {
	// Split on whitespace and common delimiters
	return tokenize(code) // Reuse for MVP
}

func computeTF(words []string) map[string]float64 {
	tf := make(map[string]float64)
	total := float64(len(words))

	if total == 0 {
		return tf
	}

	counts := make(map[string]int)
	for _, word := range words {
		counts[word]++
	}

	for word, count := range counts {
		tf[word] = float64(count) / total
	}

	return tf
}

func hashTerm(term string) int {
	h := 0
	for _, c := range term {
		h = h*31 + int(c)
	}
	if h < 0 {
		h = -h
	}
	return h
}

func normalizeCount(count, max int) float32 {
	val := float32(count) / float32(max)
	if val > 1 {
		return 1
	}
	return val
}

func avgWordLength(words []string) float32 {
	if len(words) == 0 {
		return 0
	}
	total := 0
	for _, w := range words {
		total += len(w)
	}
	return float32(total) / float32(len(words)) / 20.0 // Normalize by 20
}

func sentenceCount(text string) float32 {
	count := strings.Count(text, ".") + strings.Count(text, "!") + strings.Count(text, "?")
	return normalizeCount(count, 1000)
}

func uppercaseRatio(text string) float32 {
	upper := 0
	letters := 0
	for _, r := range text {
		if unicode.IsLetter(r) {
			letters++
			if unicode.IsUpper(r) {
				upper++
			}
		}
	}
	if letters == 0 {
		return 0
	}
	return float32(upper) / float32(letters)
}

func punctuationRatio(text string) float32 {
	punct := 0
	for _, r := range text {
		if unicode.IsPunct(r) {
			punct++
		}
	}
	return float32(punct) / float32(len(text))
}

func digitRatio(text string) float32 {
	digits := 0
	for _, r := range text {
		if unicode.IsDigit(r) {
			digits++
		}
	}
	return float32(digits) / float32(len(text))
}

func countOccurrences(text, substr string) int {
	return strings.Count(text, substr)
}

func indentationDepth(code string) float32 {
	maxDepth := 0
	for _, line := range strings.Split(code, "\n") {
		depth := 0
		for _, c := range line {
			if c == ' ' || c == '\t' {
				depth++
			} else {
				break
			}
		}
		if depth > maxDepth {
			maxDepth = depth
		}
	}
	return normalizeCount(maxDepth, 100)
}

func commentRatio(code string) float32 {
	commentChars := strings.Count(code, "//") * 2
	commentChars += strings.Count(code, "/*") * 2
	commentChars += strings.Count(code, "#") * 1
	return float32(commentChars) / float32(len(code))
}

func entropy(data []byte) float32 {
	if len(data) == 0 {
		return 0
	}

	freq := make([]int, 256)
	for _, b := range data {
		freq[b]++
	}

	entropy := 0.0
	length := float64(len(data))

	for _, count := range freq {
		if count > 0 {
			p := float64(count) / length
			entropy -= p * math.Log2(p)
		}
	}

	return float32(entropy / 8.0) // Normalize to [0, 1]
}

func countZeros(data []byte) int {
	count := 0
	for _, b := range data {
		if b == 0 {
			count++
		}
	}
	return count
}

func computeEdgeHistogram(img image.Image) []float32 {
	// Simplified edge detection (Sobel-like)
	bins := make([]float32, 64)
	// TODO: Implement proper edge detection
	return bins
}

func computeTextureFeatures(img image.Image) []float32 {
	// Simplified texture features (GLCM-like)
	features := make([]float32, 64)
	// TODO: Implement proper texture analysis
	return features
}
