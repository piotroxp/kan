# Exo Integration for Distributed Training

## Overview

**Exo** (from ExoLabs) is a distributed AI framework that supports both **training and inference** across heterogeneous devices connected via VPN. It uses dynamic model partitioning to distribute models across devices based on network topology and available resources.

## Exo Capabilities

### ✅ Supports Distributed Training
Exo is **not limited to inference** - it fully supports distributed training with:
- **Dynamic model partitioning**: Automatically distributes model layers across devices
- **Gradient synchronization**: Handles gradient aggregation across nodes
- **Parameter updates**: Coordinates parameter updates across the distributed cluster
- **Checkpoint management**: Distributed checkpoint saving/loading

### Network Discovery
- **UDP Discovery**: Automatic node discovery on local networks
- **Tailscale Discovery**: VPN-based discovery using Tailscale mesh VPN
- **Cross-network communication**: Devices in different locations can form a cluster

### API Compatibility
- **ChatGPT-compatible API**: Can integrate with existing training pipelines
- **REST API**: Standard HTTP endpoints for model operations
- **WebSocket**: Real-time communication for training coordination

---

## Integration Requirements

To integrate this KAN speech model with Exo for distributed training, we need to implement the following components:

### 1. Model Serialization/Deserialization

**Current State**: Model parameters are stored in `SpeechModel` but not fully serializable.

**Required**:
```cpp
// Add to src/model/speech_model.hpp
class SpeechModel {
public:
    // Serialize model to JSON/MessagePack for Exo
    std::string serialize() const;
    void deserialize(const std::string& data);
    
    // Get parameter count and structure
    ModelStructure get_structure() const;
    
    // Set parameters from external source (Exo)
    void set_parameters(const std::vector<std::vector<double>>& params);
};
```

**Why**: Exo needs to partition the model and send parts to different nodes. We need to serialize the model state (parameters, architecture) so Exo can:
- Split the model across devices
- Send model chunks over the network
- Reconstruct the model on receiving nodes

### 2. Gradient Aggregation Interface

**Current State**: Gradients are computed locally in `ModelBackward::backward()` but not aggregated.

**Required**:
```cpp
// Add to src/training/distributed_training.hpp
class DistributedGradientAggregator {
public:
    // Aggregate gradients from multiple nodes
    ModelBackward::Gradients aggregate(
        const std::vector<ModelBackward::Gradients>& node_grads
    );
    
    // All-reduce operation (sum gradients across nodes)
    void all_reduce_gradients(ModelBackward::Gradients& grads);
    
    // Get gradient from remote node
    ModelBackward::Gradients receive_gradient(int node_id);
    
    // Send gradient to coordinator
    void send_gradient(const ModelBackward::Gradients& grads);
};
```

**Why**: In distributed training, each node computes gradients on its local batch, then gradients must be aggregated (typically averaged) before updating parameters. Exo handles the network communication, but we need to provide the aggregation logic.

### 3. Exo API Client

**Required**:
```cpp
// New file: src/training/exo_client.hpp
class ExoClient {
public:
    // Connect to Exo cluster via Tailscale
    bool connect(const std::string& tailscale_network);
    
    // Register this node with Exo
    void register_node(const NodeCapabilities& caps);
    
    // Get model partition assignment from Exo
    ModelPartition get_partition_assignment();
    
    // Send local gradients to coordinator
    void send_gradients(const ModelBackward::Gradients& grads);
    
    // Receive aggregated gradients
    ModelBackward::Gradients receive_aggregated_gradients();
    
    // Send/receive model parameters
    void sync_parameters(const std::vector<std::vector<double>>& params);
    std::vector<std::vector<double>> receive_parameters();
    
    // Checkpoint coordination
    void save_distributed_checkpoint(const Checkpoint& ckpt);
    Checkpoint load_distributed_checkpoint();
};
```

**Why**: This is the main interface to Exo. It handles:
- Network connection (Tailscale VPN)
- Model partitioning coordination
- Gradient synchronization
- Parameter updates
- Checkpoint management

### 4. Model Partitioning Strategy

**Current State**: Model is monolithic - all layers on one device.

**Required**: Define how to split the model across nodes:

```cpp
// Model partition configuration
struct ModelPartition {
    // Which layers this node handles
    std::vector<int> layer_indices;
    
    // Input/output shapes for this partition
    std::vector<TensorShape> input_shapes;
    std::vector<TensorShape> output_shapes;
    
    // Communication requirements
    struct Communication {
        int upstream_node;      // Where inputs come from
        int downstream_node;    // Where outputs go
        size_t data_size;        // Bytes to transfer
    } comm;
};

// Partitioning strategies
class ModelPartitioner {
public:
    // Pipeline parallelism: split layers across nodes
    std::vector<ModelPartition> pipeline_partition(
        const SpeechModel& model,
        const std::vector<NodeResources>& nodes
    );
    
    // Data parallelism: same model, different data
    ModelPartition data_parallel_partition(
        const SpeechModel& model,
        int node_id,
        int total_nodes
    );
};
```

**KAN Model Partitioning Considerations**:
- **Pipeline Parallelism**: Split by stage:
  - Node 1: Audio → Mel-spectrogram (SincKAN)
  - Node 2: Mel-spectrogram → Quantum Embeddings (Chebyshev KAN)
  - Node 3: Quantum → Semantic (B-spline KAN)
  - Node 4: Semantic → Classification (B-spline KAN)
- **Data Parallelism**: Each node processes different batches, gradients aggregated
- **Hybrid**: Combine both approaches

### 5. Training Loop Modifications

**Current State**: `TrainingSession::train_epoch()` processes batches locally.

**Required**: Modify to support distributed training:

```cpp
// Modified training_session.hpp
class DistributedTrainingSession {
public:
    DistributedTrainingSession(
        const std::string& exo_endpoint,
        const std::string& tailscale_network
    );
    
    void train() {
        // Connect to Exo cluster
        exo_client_.connect(tailscale_network);
        
        // Get model partition assignment
        auto partition = exo_client_.get_partition_assignment();
        model_.set_partition(partition);
        
        // Distributed training loop
        for (size_t epoch = 0; epoch < num_epochs_; ++epoch) {
            train_epoch_distributed(epoch);
        }
    }
    
private:
    void train_epoch_distributed(size_t epoch) {
        // Each node processes its local batch
        auto batch = batch_generator_.generate_synthetic_batch();
        auto output = model_.forward_partitioned(batch.audio);
        
        // Compute local gradients
        auto grads = ModelBackward::backward(output, loss_grad, model_);
        
        // Send gradients to coordinator
        exo_client_.send_gradients(grads);
        
        // Receive aggregated gradients
        auto aggregated_grads = exo_client_.receive_aggregated_gradients();
        
        // Update local parameters
        update_parameters_with_gradients(aggregated_grads);
        
        // Sync parameters across nodes (periodic)
        if (step_ % sync_frequency_ == 0) {
            exo_client_.sync_parameters(model_.get_parameters());
        }
    }
    
    ExoClient exo_client_;
    int sync_frequency_ = 10;  // Sync every 10 steps
};
```

### 6. Network Communication Layer

**Required**: HTTP/WebSocket client for Exo API:

```cpp
// New file: src/training/exo_network.hpp
class ExoNetworkClient {
public:
    // HTTP POST to Exo API
    std::string post(const std::string& endpoint, const std::string& data);
    
    // WebSocket connection for real-time updates
    void connect_websocket(const std::string& url);
    void send_message(const std::string& message);
    std::string receive_message();
    
    // Tailscale integration
    bool is_tailscale_connected();
    std::vector<std::string> discover_nodes();
};
```

**Dependencies Needed**:
- HTTP client library (e.g., `libcurl`, `httplib`)
- WebSocket library (e.g., `websocketpp`)
- JSON serialization (e.g., `nlohmann/json`)

### 7. Checkpoint Coordination

**Current State**: Checkpoints saved locally.

**Required**: Distributed checkpoint management:

```cpp
// Modified checkpoint_manager.hpp
class DistributedCheckpointManager {
public:
    // Save checkpoint to Exo cluster (replicated)
    void save_distributed(
        const std::string& name,
        const Checkpoint& ckpt
    );
    
    // Load checkpoint from Exo (from any node)
    Checkpoint load_distributed(const std::string& name);
    
    // Coordinate checkpoint saving (only coordinator saves)
    void save_if_coordinator(const std::string& name, const Checkpoint& ckpt);
};
```

---

## Infrastructure Requirements

### VPN Setup (Tailscale)

1. **Install Tailscale** on all training nodes:
```bash
# On each node
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```

2. **Configure Tailscale network**:
   - Create Tailscale network/ACL
   - Ensure all nodes can communicate
   - Set up subnets if needed

3. **Verify connectivity**:
```bash
tailscale status  # List all nodes
ping <node-ip>     # Test connectivity
```

### Exo Installation

1. **Install Exo** on coordinator node:
```bash
# Follow ExoLabs installation instructions
# Typically involves:
# - Installing Exo daemon
# - Configuring Exo API endpoint
# - Setting up authentication
```

2. **Configure Exo**:
   - Set Tailscale network for discovery
   - Configure model partitioning strategy
   - Set up checkpoint storage (shared or replicated)

### Node Requirements

Each training node needs:
- **Network**: Tailscale VPN access
- **Compute**: CPU or GPU (AMD ROCm for GPU nodes)
- **Memory**: Sufficient for model partition + batch data
- **Storage**: Local checkpoint storage (optional if using shared storage)
- **Software**: 
  - Exo client library
  - This KAN model codebase
  - Dependencies (ROCm, Conan packages)

### Coordinator Node

One node acts as coordinator:
- **Exo server**: Runs Exo daemon
- **Gradient aggregation**: Collects and aggregates gradients
- **Parameter synchronization**: Coordinates parameter updates
- **Checkpoint management**: Saves/loads checkpoints

---

## Implementation Plan

### Phase 1: Basic Integration (2-3 weeks)

1. **Model Serialization** (Week 1)
   - Implement `serialize()`/`deserialize()` in `SpeechModel`
   - Add parameter getter/setter methods
   - Test serialization round-trip

2. **Exo API Client** (Week 1-2)
   - Implement HTTP client for Exo API
   - Add Tailscale discovery
   - Test connection to Exo cluster

3. **Gradient Aggregation** (Week 2)
   - Implement gradient aggregation logic
   - Add all-reduce operation
   - Test with synthetic gradients

### Phase 2: Distributed Training (2-3 weeks)

4. **Model Partitioning** (Week 3)
   - Implement pipeline parallelism
   - Implement data parallelism
   - Test partitioning strategies

5. **Distributed Training Loop** (Week 3-4)
   - Modify `TrainingSession` for distributed mode
   - Integrate Exo client
   - Test with 2-4 nodes

6. **Checkpoint Coordination** (Week 4)
   - Implement distributed checkpoint saving
   - Add checkpoint loading from coordinator
   - Test checkpoint resumption

### Phase 3: Optimization (1-2 weeks)

7. **Performance Optimization** (Week 5)
   - Optimize network communication (batch gradients)
   - Implement gradient compression
   - Profile and optimize bottlenecks

8. **Fault Tolerance** (Week 5-6)
   - Add node failure handling
   - Implement checkpoint-based recovery
   - Test fault scenarios

---

## Example Usage

### Single-Node Training (Current)
```bash
./train 16 1e-4 50
```

### Distributed Training with Exo
```bash
# On coordinator node
export EXO_COORDINATOR=true
export TAILSCALE_NETWORK=my-network
./train --distributed --exo-endpoint=http://exo:8080 16 1e-4 50

# On worker nodes
export EXO_WORKER=true
export EXO_COORDINATOR_URL=http://coordinator:8080
export TAILSCALE_NETWORK=my-network
./train --distributed --exo-endpoint=http://coordinator:8080 16 1e-4 50
```

---

## Challenges and Considerations

### 1. Network Latency
- **Issue**: VPN latency can slow gradient synchronization
- **Solution**: 
  - Use gradient accumulation to reduce sync frequency
  - Compress gradients before transmission
  - Use async parameter updates (stale gradients)

### 2. Model Partitioning Overhead
- **Issue**: Splitting model adds communication overhead
- **Solution**:
  - Use data parallelism for small models
  - Pipeline parallelism for large models
  - Hybrid approach based on model size

### 3. Heterogeneous Hardware
- **Issue**: Nodes may have different compute capabilities
- **Solution**:
  - Exo's dynamic partitioning handles this
  - Assign larger partitions to more powerful nodes
  - Use load balancing

### 4. Checkpoint Consistency
- **Issue**: Ensuring all nodes have consistent model state
- **Solution**:
  - Coordinator saves checkpoints
  - Periodic parameter synchronization
  - Version numbers for checkpoints

### 5. Quantum Embedding Complexity
- **Issue**: Complex-valued wavefunctions need special handling
- **Solution**:
  - Serialize as real/imaginary pairs
  - Ensure fidelity computation is consistent across nodes

---

## Testing Strategy

1. **Unit Tests**:
   - Model serialization/deserialization
   - Gradient aggregation correctness
   - Partitioning logic

2. **Integration Tests**:
   - 2-node distributed training
   - Gradient synchronization
   - Checkpoint save/load

3. **End-to-End Tests**:
   - Multi-node training (4+ nodes)
   - Fault tolerance (node failure)
   - Performance benchmarks

---

## References

- **Exo Documentation**: [ExoLabs Exo](https://deepwiki.com/exo-explore/exo)
- **Tailscale**: [Tailscale Documentation](https://tailscale.com/kb/)
- **Distributed Training**: PyTorch DDP, Horovod patterns

---

## Summary

**Exo supports distributed training**, not just inference. To integrate:

1. ✅ **Model serialization** - Serialize/deserialize model state
2. ✅ **Gradient aggregation** - Aggregate gradients across nodes
3. ✅ **Exo API client** - Connect to Exo cluster via Tailscale
4. ✅ **Model partitioning** - Split model across nodes (pipeline/data parallel)
5. ✅ **Distributed training loop** - Modify training to use Exo
6. ✅ **Checkpoint coordination** - Distributed checkpoint management
7. ✅ **Network layer** - HTTP/WebSocket client for Exo API

**Estimated effort**: 5-6 weeks for full integration with testing.

**Infrastructure**: Tailscale VPN + Exo coordinator + worker nodes with ROCm (for GPU nodes).
