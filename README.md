# TERA

**Terrestrial Extendable Retrieval & Addressing**

A semantic content discovery network with cryptographic integrity guarantees.

## What is TERA?

TERA is a peer-to-peer network that enables similarity-based content discovery while preventing spam through cryptographic verification. Unlike traditional content-addressed networks (like IPFS) that require exact content hashes, TERA allows you to find "similar" content with provable integrity.

## The Core Innovation

Traditional DHTs have a fundamental limitation: they can only verify exact content identity. TERA introduces a **dual-output hash function** that provides:

1. **H_crypto** - A homomorphic hash supporting O(1) incremental extensions
2. **H_semantic** - Universal neural kernels for multi-modal similarity

This enables **integrity-gated semantic search**: nodes can verify that content legitimately extends a root hash while filtering spam based on relevance.

### Key Properties

- **Universal**: Works with any content type (text, images, code, audio, binary)
- **Extendable**: Add content to a collection in O(1) time without recomputing the entire hash
- **Verifiable**: Cryptographically prove that content B extends content A
- **Spam-resistant**: Invalid extensions are automatically rejected by the network
- **Parameterized**: Users define their own notion of "similarity" via runtime kernel parameters
- **Model-agnostic**: Neural kernels are content-addressed and exchangeable (like codecs)

## How It Works

```
Traditional IPFS:
Content → SHA-256 → Exact lookup → Retrieve

TERA:
Content → (H_crypto, H_semantic) → Similarity search + Verification → Discover
```

**Example:**

```go
// Extract universal features (works for text, images, code, etc.)
features, _ := semantic.ExtractFeatures(content, filename)
// → FeatureVector{Modality: "text", Data: [512]float32, Hash: "..."}

// Load a neural kernel by content ID (cached locally like a codec)
registry := semantic.NewKernelRegistry("~/.tera/kernels")
kernel, _ := registry.Get("bafyreisemantic...")  // IPLD CID

// Compute similarity with runtime parameters
params := semantic.KernelParams{
    WeightSemantic:   0.7,
    WeightLexical:    0.3,
    WeightStructural: 0.1,
    Threshold:        0.6,
}
similarity, _ := kernel.ComputeSimilarity(featuresA, featuresB, params)

// Create extendable content with cryptographic proof
root := tera.NewContent(features)
extended := root.Extend(newFeatures)  // O(1) verification

// Network forwards only if:
// 1. H_crypto verifies (legitimate extension)
// 2. Neural kernel similarity exceeds threshold (relevant)
```

## Neural Kernels: The Codec Analogy

TERA solves the heterogeneous model problem in distributed systems. In a P2P network, different nodes may use different LLMs (Claude, GPT-4, local models), which produce incompatible embeddings. TERA's solution:

**Content-Addressed Kernels** (like audio/video codecs):
- Small neural networks (100KB-1MB) for computing similarity
- Referenced by IPLD CID (e.g., `bafyreisemantic...`)
- Cached locally, downloaded on-demand
- Multiple kernels coexist (like having H.264, VP9, AV1)

**Universal Features** (model-agnostic):
- Fixed-size vectors (512 floats) for all content types
- Extracted deterministically (no training needed)
- Text: TF-IDF + n-grams, Images: color/edge histograms, Code: token frequency
- Lightweight enough to transmit over network

**Runtime Parameters** (high dexterity):
- Users tune kernel behavior at query time
- No need to recompute features or retrain models
- Example: Adjust semantic vs lexical focus per query

This design means nodes can share a "taste" (kernel) without forcing everyone to use the same LLM.

## Architecture

```
┌─────────────────────────────────────────┐
│ Application Layer                       │
│ (Queries, Publications, Subscriptions)  │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│ Gossip Protocol + Gatekeeping           │
│ (Forward if: valid_extension ∧ similar) │
└─────────────────────────────────────────┘
                  ↓
┌──────────────────┬──────────────────────┐
│ H_crypto         │ H_semantic           │
│ (Homomorphic)    │ (Neural Kernels)     │
│ Integrity ✓      │ Discovery ✓          │
└──────────────────┴──────────────────────┘
                  ↓
┌──────────────────┬──────────────────────┐
│ Storage Layer    │ Kernel Registry      │
│ (BadgerDB)       │ (IPLD Cache)         │
└──────────────────┴──────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│ libp2p Transport Layer                  │
└─────────────────────────────────────────┘
```

## Project Status

**Phase 1: Primitives** (Complete ✓)
- [x] Homomorphic hash implementation (`crypto/`)
- [x] Universal feature extraction (`semantic/features_universal.go`)
- [x] Native neural kernels (`semantic/neural.go`)
- [x] IPLD kernel descriptors (`semantic/kernel_model.go`)
- [x] Gatekeeping logic (`core/`)
- [x] Working demo (`examples/demo.go`)

**Phase 2: Network** (Complete ✓)
- [x] libp2p integration (`network/`)
- [x] Gossip protocol with pubsub
- [x] Basic node implementation
- [x] CLI tool (`tera-node`)

**Phase 3: Storage & Kernels** (In Progress)
- [x] Extension graph storage (`storage/`)
- [x] BadgerDB persistence layer
- [x] Kernel registry with caching
- [x] Multi-modal feature extraction
- [ ] Content discovery API
- [ ] Interest subscription mechanism
- [ ] IPFS integration for content
- [ ] DHT for peer discovery
- [ ] Kernel training utilities

## Quick Start

```bash
# Run tests
go test ./...

# Run demo (local simulation)
go run examples/demo.go

# Build and run network node
go build -o tera-node ./cmd/tera-node

# Start bootstrap node
./tera-node -port 9000 -interests "machine learning"

# Start second node (in another terminal)
# Replace <PEER-ID> with the peer ID from bootstrap node
./tera-node -port 9001 \
  -bootstrap "/ip4/127.0.0.1/tcp/9000/p2p/<PEER-ID>" \
  -interests "machine learning"

# In the node shell, publish content:
> publish neural networks and deep learning algorithms
> stats
> peers
```

## Why "Terrestrial"?

It's a pun. IPFS is "InterPlanetary" File System, so naturally this is the "Terrestrial" version. Supposedly inverted priorities: we're focused on Earth-based problems like spam, discovery, and semantic search rather than interplanetary data transfer.

## Related Work

- **IPFS**: Content-addressed storage (exact matching only)
- **DHT2VEC** (2020): Early exploration of semantic DHTs
- **Kademlia**: XOR-metric DHT routing (no semantic awareness)

TERA combines ideas from content-addressed networks, homomorphic cryptography, and kernel methods to create something new: **semantic content addressing with integrity**.

## License

BSD 3-Clause License (see LICENSE file)

## Contributing

This project is in early development. Contributions welcome once the core primitives are stable.
