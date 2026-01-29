// Package semantic - Compatibility layer for the new universal feature API
// This file provides backward-compatible types and functions used by core/ and network/
package semantic

import (
	"math"
	"sort"
)

// Features is a type alias for backward compatibility.
// The new code uses FeatureVector, but existing code references Features.
type Features = FeatureVector

// ExtractFeaturesSimple extracts features without requiring a filename.
// This is a convenience wrapper for code that doesn't have filename context.
func ExtractFeaturesSimple(content []byte) *Features {
	features, err := ExtractFeatures(content, "")
	if err != nil {
		// Return empty features on error (maintains backward compatibility)
		return &Features{
			Modality: "unknown",
			Size:     512,
			Data:     NewVector(512),
			Hash:     "",
		}
	}
	return features
}

// Similarity computes similarity between two feature vectors using the kernel parameters.
// Returns a value in [0, 1] where 1 means identical.
func Similarity(a, b *Features, params KernelParams) float64 {
	if a == nil || b == nil {
		return 0.0
	}

	// Handle empty vectors
	if len(a.Data) == 0 || len(b.Data) == 0 {
		return 0.0
	}

	// Compute similarity using multiple signals
	// 1. Term overlap (Jaccard-like on non-zero positions) - captures shared vocabulary
	// 2. Cosine similarity on term frequencies - captures relative term importance
	// 3. Statistical similarity - captures document structure

	termSim := termOverlapSimilarity(a.Data, b.Data)
	cosineSim := cosineSimilarityOnTerms(a.Data, b.Data)

	// Combine with weights from params
	// The Jaccard-like term overlap is more discriminative for text
	similarity := params.WeightSemantic*cosineSim + params.WeightLexical*termSim

	// Normalize by total weight
	totalWeight := params.WeightSemantic + params.WeightLexical
	if totalWeight > 0 {
		similarity /= totalWeight
	}

	// Apply temperature scaling
	if params.Temperature > 0 && params.Temperature != 1.0 {
		similarity = math.Pow(similarity, 1.0/params.Temperature)
	}

	// Clamp to [0, 1]
	if similarity < 0 {
		similarity = 0
	}
	if similarity > 1 {
		similarity = 1
	}

	return similarity
}

// termOverlapSimilarity computes Jaccard-like similarity based on non-zero term positions.
// This is more discriminative for text with different vocabularies.
func termOverlapSimilarity(a, b Vector) float64 {
	// Only look at the term-hash part (exclude statistical features at end)
	termLen := len(a) - 16
	if termLen < 0 {
		termLen = len(a)
	}
	if termLen > len(b) {
		termLen = len(b) - 16
		if termLen < 0 {
			termLen = len(b)
		}
	}

	threshold := float32(0.001) // Consider position "active" if > threshold
	var intersection, union int

	for i := 0; i < termLen; i++ {
		aActive := a[i] > threshold
		bActive := b[i] > threshold

		if aActive || bActive {
			union++
		}
		if aActive && bActive {
			intersection++
		}
	}

	if union == 0 {
		return 0.0
	}

	return float64(intersection) / float64(union)
}

// cosineSimilarityOnTerms computes cosine similarity only on the term-hash part of vectors.
func cosineSimilarityOnTerms(a, b Vector) float64 {
	// Only look at the term-hash part (exclude statistical features at end)
	termLen := len(a) - 16
	if termLen < 0 {
		termLen = len(a)
	}
	if termLen > len(b)-16 {
		termLen = len(b) - 16
		if termLen < 0 {
			termLen = len(b)
		}
	}

	var dot, normA, normB float64
	for i := 0; i < termLen; i++ {
		dot += float64(a[i] * b[i])
		normA += float64(a[i] * a[i])
		normB += float64(b[i] * b[i])
	}

	if normA == 0 || normB == 0 {
		return 0.0
	}

	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

// IsRelevant checks if two feature vectors are similar enough given the threshold.
func IsRelevant(a, b *Features, params KernelParams) bool {
	sim := Similarity(a, b, params)
	return sim >= params.Threshold
}

// RankedResult represents a similarity search result with its score.
type RankedResult struct {
	Index      int
	Features   *Features
	Similarity float64
}

// RankBySimilarity ranks a list of features by similarity to a query.
// Returns results sorted by similarity in descending order.
func RankBySimilarity(query *Features, features []*Features, params KernelParams) []RankedResult {
	results := make([]RankedResult, 0, len(features))

	for i, f := range features {
		if f == nil {
			continue
		}
		sim := Similarity(query, f, params)
		results = append(results, RankedResult{
			Index:      i,
			Features:   f,
			Similarity: sim,
		})
	}

	// Sort by similarity descending
	sort.Slice(results, func(i, j int) bool {
		return results[i].Similarity > results[j].Similarity
	})

	return results
}
