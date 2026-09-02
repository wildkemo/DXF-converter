# Graph Transformer for DXF Correction — Implementation Plan

## 1. Objective

Build a Graph Transformer that learns:

Corrupted DXF Graph → Correct DXF Graph

The DXF files are already available as pairs, so the focus here is the graph representation, model architecture, training, and inference.

The model should eventually handle:

- coordinate inaccuracies
- jagged lines
- misaligned geometry
- incorrect connections
- extra entities
- missing entities
- incorrect entity types

However, implement these progressively.

## 2. Start With the Right Problem Formulation

Do not initially try to make the Transformer generate an entire DXF from scratch.

Instead formulate the first version as:

Input graph
↓
Graph Transformer
↓
Corrected properties for each input node

For example:

Corrupted:

A: LINE
(0.03, 0.02) → (4.98, 0.05)

Target:

A: LINE
(0.00, 0.00) → (5.00, 0.00)

The model learns:

[0.03, 0.02, 4.98, 0.05]
↓
Graph Transformer
↓
[0.00, 0.00, 5.00, 0.00]

This is much easier than asking the model to generate arbitrary DXF entities.

## 3. Graph Definition

Represent the DXF as a graph:

G = (V, E)

Where:

V = geometric entities
E = relationships between entities

For example:

       wall1
      /     \

wall2 wall4
\ /
wall3

Each wall is a node.

## 4. Node Definition

Initially:

1 DXF geometric entity = 1 graph node

Examples:

- LINE
- LWPOLYLINE
- ARC
- CIRCLE

Each node should contain:

- entity type
- geometry
- position
- size
- orientation

## 5. Node Feature Vector

For a line:

- x1
- y1
- x2
- y2
- length
- angle
- center_x
- center_y
- bbox_width
- bbox_height

The raw feature vector could therefore be:

[
x1,
y1,
x2,
y2,
length,
angle,
center_x,
center_y,
bbox_width,
bbox_height
]

Add the entity type separately using an embedding.

For example:

LINE → embedding 0
LWPOLYLINE → embedding 1
ARC → embedding 2
CIRCLE → embedding 3

Do not simply encode the type as:

LINE = 0
ARC = 1
CIRCLE = 2

and expect the network to understand the categorical meaning.

Use a learned embedding.

## 6. Coordinate Normalization

Normalize the geometry before feeding it to the Transformer.

For example:

x' = (x - min_x) / (max_x - min_x)
y' = (y - min_y) / (max_y - min_y)

Therefore:

x' ∈ [0,1]
y' ∈ [0,1]

This prevents the model from having to learn that:

1000 → 1050

and:

0.1 → 0.15

can represent the same relative geometric correction.

Store the original bounding box so predictions can later be converted back.

## 7. Edge Definition

Edges represent geometric relationships.

Useful relationships include:

- nearby
- touching
- intersecting
- connected
- parallel
- perpendicular
- same-layer

For the first version, don't try to encode every possible relationship.

Start with:

- spatial proximity
- endpoint proximity

## 8. Building the Graph

For every node:

Find nearby nodes.
Connect the closest K nodes.
Add explicit edges where entities physically touch/intersect.

Start with:

K = 8

For example:

             N2
             |
             |

N1 ----------N3---------- N4
|
|
N5

The resulting graph allows the Transformer to understand local geometric context.

## 9. Edge Features

Every edge should contain geometric information.

For nodes i and j:

- distance
- relative_x
- relative_y
- angle
- endpoint_distance
- intersection
- parallel
- perpendicular

Example:

edge(i,j) =
[
distance,
relative_x,
relative_y,
angle_difference,
endpoint_distance,
intersects,
parallel,
perpendicular
]

Normalize continuous values.

Boolean relationships can be represented as:

0 / 1

## 10. Why Edge Features Matter

Consider:

A ───────── B

and:

A
|
|
B

The node features tell the model where A and B are.

The edge features tell the model:

- A and B are close
- A and B are aligned
- A and B are perpendicular

This information is extremely useful for floorplan correction.

## 11. Graph Transformer Input

Each training example should eventually look approximately like:

Data(
x=node_features,
edge_index=edge_index,
edge_attr=edge_features,
y=target_geometry
)

Where:

x
→ corrupted node features

edge_index
→ graph connectivity

edge_attr
→ geometric relationship information

y
→ corresponding correct geometry

## 12. Entity Matching

This is critical.

The model needs to know which corrupted entity corresponds to which perfect entity.

If:

corrupted:

N0
N1
N2
N3

and:

perfect:

N0
N1
N2
N3

the correspondence is easy.

But DXF entity ordering might instead be:

corrupted:

N0
N1
N2
N3

perfect:

N3
N0
N2
N1

Therefore the training pipeline needs correspondence information.

## 13. Initial Matching Strategy

For every corrupted entity, find the corresponding target entity using:

- entity type
- spatial distance
- length similarity
- orientation similarity
- neighbor similarity

A simple matching score:

score =
w_position \* position_distance

- w_length \* length_difference
- w_angle \* angle_difference
- w_type \* type_difference

Choose the lowest-cost matching target.

For difficult cases, use a global matching algorithm such as the Hungarian algorithm instead of independently choosing the nearest entity.

## 14. First Model: Coordinate Correction

Do not implement missing/extra entity generation initially.

The first model should assume:

- same entities
- same entity types
- same topology

Only coordinates are corrupted.

The architecture becomes:

Corrupted Graph
↓
Node Embedding
↓
Graph Transformer
↓
Node Embeddings
↓
Coordinate Head
↓
Correct Coordinates

## 15. Model Architecture

Use:

Input dimension
↓
Linear projection
↓
Hidden dimension
↓
Graph Transformer × 4
↓
Node representation
↓
MLP prediction head
↓
Corrected geometry

Initial configuration:

- hidden_dim = 256
- layers = 4
- attention_heads = 8
- dropout = 0.1

Do not immediately make the model huge.

A small model is easier to debug and should be sufficient for an initial experiment.

## 16. Node Embedding

The raw geometry should first be projected into the Transformer dimension.

Conceptually:

geometry features
↓
Linear
↓
256-dimensional vector

Entity type should have its own embedding:

entity_type
↓
Embedding
↓
type vector

Then combine:

node_embedding =
geometry_embedding +
type_embedding

## 17. Spatial Encoding

Normal Transformer positional encoding is not enough.

The model needs 2D spatial information.

The node embedding should contain:

- absolute position
- relative positions
- distance relationships
- orientation

Edge attributes are especially important here.

Think of it as:

Transformer attention +
geometric relationship information

rather than a normal sequence Transformer.

## 18. Graph Attention

The Transformer should allow every node to look at its graph neighbors.

For example:

      wall B
         │
         │

wall A ──┼── wall C
│
│
wall D

When correcting wall A, the model can use:

- wall B
- wall C
- wall D

to determine what the intended geometry probably was.

This is the major advantage over correcting each DXF entity independently.

## 19. Message Passing

Conceptually, each Transformer layer performs:

node
↓
look at neighboring nodes
↓
calculate attention
↓
combine information
↓
produce updated node representation

After several layers:

Layer 1
→ local relationships

Layer 2
→ slightly larger neighborhood

Layer 3
→ broader geometry

Layer 4
→ global/local structural context

## 20. Coordinate Prediction Head

For a LINE:

Transformer embedding
↓
Linear
↓
ReLU
↓
Linear
↓
4 values

Output:

x1'
y1'
x2'
y2'

For example:

Input:

0.03 0.02 0.49 0.05

Output:

0.00 0.00 0.50 0.00

## 21. Prediction Residual vs Absolute Coordinates

Prefer predicting a correction delta initially.

Instead of:

model → target_coordinates

use:

model → delta

Then:

corrected = corrupted + delta

Example:

corrupted:
0.03, 0.02

model:
-0.03, -0.02

corrected:
0.00, 0.00

This is usually a better formulation because the corrupted geometry is already close to the target.

## 22. Residual Prediction

For a LINE:

input:

x1
y1
x2
y2

Model predicts:

Δx1
Δy1
Δx2
Δy2

Then:

x1_corrected = x1 + Δx1
y1_corrected = y1 + Δy1

x2_corrected = x2 + Δx2
y2_corrected = y2 + Δy2

This should be your initial coordinate-correction formulation.

## 23. Loss Function

Start with:

Smooth L1 Loss

between:

predicted corrected coordinates

and:

perfect coordinates

Or equivalently between:

predicted delta

and:

target delta

Use the delta formulation if that's how the model is implemented.

## 24. Why Not MSE Initially?

MSE heavily penalizes large errors.

DXF corruption can contain occasional large coordinate deviations.

Smooth L1 behaves more gently:

- small error
  → behaves approximately like MSE

- large error
  → behaves approximately like L1

This makes it a good starting loss.

## 25. Training

Training looks like:

corrupted graph
↓
Graph Transformer
↓
predicted corrections
↓
compare with perfect graph
↓
loss
↓
backpropagation
↓
update weights

Repeat this over thousands of graph pairs.

## 26. Training Loop

Conceptually:

for batch in train_loader:

    predicted_delta = model(batch)

    loss = loss_function(
        predicted_delta,
        batch.target_delta
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

The important operations are:

zero_grad()
→ remove gradients from previous iteration

backward()
→ calculate how each parameter contributed to the error

step()
→ update model parameters

## 27. Optimizer

Start with:

AdamW

Use:

- learning_rate = 1e-4
- weight_decay = 1e-4

Monitor validation loss.

If training is unstable:

reduce learning rate

If training is extremely slow:

try 2e-4

Do not blindly change many parameters simultaneously.

## 28. Batch Training

Use PyTorch Geometric's graph batching.

You can have:

Graph 1 → 40 nodes
Graph 2 → 120 nodes
Graph 3 → 65 nodes

in the same batch.

PyTorch Geometric combines them into one disconnected graph internally.

The model can therefore process drawings of different sizes.

## 29. Variable Number of Entities

This is one of the reasons to use a Graph Transformer.

You don't need:

exactly 100 entities per DXF

A graph can contain:

20 nodes

or:

500 nodes

and the same model can process both.

## 30. Handling LWPOLYLINE

Do not immediately make a complicated variable-length decoder.

Initially convert a polyline into individual segments:

P1 → P2
P2 → P3
P3 → P4

Therefore:

POLYLINE

becomes multiple graph nodes.

This simplifies the prediction problem considerably.

Later you can introduce a dedicated polyline representation if needed.

## 31. Handling Arcs and Circles

Use entity-specific geometry.

For an ARC:

- center_x
- center_y
- radius
- start_angle
- end_angle

Predict:

- Δcenter_x
- Δcenter_y
- Δradius
- Δstart_angle
- Δend_angle

For a CIRCLE:

- center_x
- center_y
- radius

predict:

- Δcenter_x
- Δcenter_y
- Δradius

## 32. Entity-Type-Specific Heads

The Transformer itself can be shared.

Only the final prediction heads differ:

                 Transformer
                      │
          ┌───────────┼───────────┐
          ↓           ↓           ↓
      LINE head    ARC head    CIRCLE head
          ↓           ↓           ↓
      LINE Δ       ARC Δ       CIRCLE Δ

This lets the Transformer learn general geometric reasoning while each head handles entity-specific geometry.

## 33. Geometric Constraint Loss

Once coordinate correction works, add geometry-aware losses.

For example, if two walls should connect:

endpoint_A ≈ endpoint_B

Add:

L_connection

If two walls should be perpendicular:

angle_A ≈ angle_B + 90°

Add:

L_perpendicular

If walls should be parallel:

angle_A ≈ angle_B

Add:

L_parallel

## 34. Final Loss

Eventually:

L_total =
λ_coord \* L_coord

- λ_connection \* L_connection
- λ_parallel \* L_parallel
- λ_perpendicular \* L_perpendicular
- λ_topology \* L_topology

Initially:

λ_coord = 1

and everything else:

0

Then introduce the additional losses gradually.

## 35. Topology Prediction

After coordinate correction works, teach the model to understand topology.

For each pair of related nodes predict:

- connected
- not connected
- parallel
- perpendicular
- intersecting

Use the Transformer embeddings:

h_i
h_j

Combine them:

[h_i, h_j, |h_i-h_j|]

and feed them into an MLP.

Output:

relationship probabilities

## 36. Why Topology Should Come Later

Trying to solve:

- coordinates
- topology
- missing entities
- extra entities

from the beginning makes debugging extremely difficult.

First prove:

Graph Transformer
→ coordinate correction

Then expand it.

## 37. Extra Entity Handling

Once the basic model works, add an existence head.

For every corrupted node:

- KEEP
- DELETE

The model predicts:

P(keep)
P(delete)

Training target:

1 = entity exists in perfect drawing
0 = entity should be removed

## 38. Missing Entities

Missing entities are fundamentally different.

If an entity doesn't exist in the corrupted graph, there is no input node from which to predict its coordinates.

Therefore you need a graph decoder capable of generating new nodes.

Eventually:

Corrupted Graph
↓
Graph Transformer Encoder
↓
Latent representation
↓
Graph Decoder
↓
Generate entities

Do not implement this until the simpler model is working.

## 39. Recommended Final Architecture

The mature system should look like:

                  Corrupted Graph
                         │
                         ▼
                Node Feature Encoder
                         │
                         ▼
                Edge Feature Encoder
                         │
                         ▼
              ┌─────────────────────┐
              │  Graph Transformer  │
              │                     │
              │  Layer 1            │
              │  Layer 2            │
              │  Layer 3            │
              │  Layer 4            │
              └──────────┬──────────┘
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
    Coordinate       Topology       Existence
       Head             Head           Head
          │              │              │
          ▼              ▼              ▼
       ΔGeometry     Relations      Keep/Delete
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                  Graph Decoder
                         │
                         ▼
                 Corrected Graph

## 40. Training Stages

### Stage 1

Implement:

- LINE entities only

Train:

- coordinate correction

Goal:

jagged lines → straight/correct lines

### Stage 2

Add:

- LWPOLYLINE

Train coordinate correction.

### Stage 3

Add:

- ARC
- CIRCLE

Use entity-specific prediction heads.

### Stage 4

Add:

- connection loss
- parallel loss
- perpendicular loss

### Stage 5

Add:

- topology prediction

### Stage 6

Add:

- extra entity detection

### Stage 7

Implement:

- missing entity generation

This is where the architecture becomes an actual graph-to-graph model rather than simply a graph regression model.

## 41. Evaluation

Do not evaluate only using training loss.

Measure:

- Coordinate Error
  - average Euclidean distance
  - between predicted and target coordinates
- Endpoint Error
  - average distance between
  - predicted and target endpoints
- Angular Error
  - average difference in orientation
- Connectivity Accuracy
  - correct connections / total connections
- Entity Existence Accuracy
  - correct KEEP/DELETE predictions

## 42. Most Important Evaluation

Always render:

CORRUPTED PREDICTED PERFECT

side by side.

For example:

┌───────────┬───────────┬───────────┐
│ corrupted │ predicted │ perfect │
│ │ │ │
│ jagged │ corrected │ reference │
└───────────┴───────────┴───────────┘

A model can have a low numerical loss while producing geometry that still looks bad.

## 43. Inference

Once trained:

corrupted.dxf
↓
DXF → Graph
↓
normalize
↓
Graph Transformer
↓
predicted Δgeometry
↓
corrupted geometry + Δgeometry
↓
denormalize
↓
corrected geometry
↓
DXF writer
↓
corrected.dxf

The perfect DXF is not used during inference.

## 44. Important Design Decision

Keep these responsibilities separate:

- DXF Parser
  - handles DXF
- Graph Builder
  - handles geometry → graph
- Graph Transformer
  - handles learned correction
- Geometry Decoder
  - handles model output → geometry
- DXF Writer
  - handles geometry → DXF

Do not put DXF-specific logic inside the neural network.

## 45. Recommended First Milestone

Your first working model should be exactly:

                 corrupted.dxf
                       ↓
                  DXF parser
                       ↓
                  Graph builder
                       ↓
              Node + edge features
                       ↓
              Graph Transformer
                       ↓
              LINE coordinate head
                       ↓
                 Δcoordinates
                       ↓
             corrected coordinates
                       ↓
                  DXF writer
                       ↓
                 predicted.dxf

Dataset restriction:

- same entities
- same topology
- only coordinates are corrupted

If this works reliably, you have proven the fundamental architecture.

Then progressively add:

geometry constraints
↓
topology
↓
extra entities
↓
missing entities

This progression is much safer than attempting the complete DXF-to-DXF correction problem with one giant model from the beginning.
