# Continuous Learning System Flow

## System Architecture

```mermaid
graph TB
    subgraph "Production Inference"
        A[User Query] --> B[Model Inference]
        B --> C[Response]
        C --> D[Log to SQLite]
        D --> E[Record Feedback]
    end

    subgraph "Usage Database"
        D --> F[(usage.db)]
        E --> F
        F --> |queries| G[Quality Score]
        F --> |feedback| H[User Feedback +1/-1]
    end

    subgraph "Continuous Learning Loop"
        I[Check Triggers<br/>Every Hour] --> J{Trigger?}
        J -->|Yes| K[Generate Training Data]
        J -->|No| I
        K --> L[Filter by Quality ≥0.7]
        L --> M[Filter by Feedback ≥1]
        M --> N[Deduplicate]
        N --> O[Export JSONL]
        O --> P[Train Model]
        P --> Q[Save Weights]
        Q --> R[Deploy as Challenger]
    end

    subgraph "A/B Testing"
        R --> S[10% Traffic]
        T[Champion Model] --> U[90% Traffic]
        S --> V[Compare Metrics<br/>48 hours]
        U --> V
        V --> W{Challenger<br/>5% Better?}
        W -->|Yes| X[Promote to Champion]
        W -->|No| Y[Keep Testing]
        X --> T
    end

    F -.-> I
    T --> B
    R --> B
```

## Trigger Decision Flow

```mermaid
graph TD
    A[Check Triggers] --> B{In Cooldown?}
    B -->|Yes| Z[Skip Retrain]
    B -->|No| C{Sample Count<br/>≥1000?}
    C -->|Yes| D[Trigger: Sample Count]
    C -->|No| E{Weekly<br/>Schedule?}
    E -->|Yes| F[Trigger: Scheduled]
    E -->|No| G{Quality Drop<br/>>10%?}
    G -->|Yes| H[Trigger: Quality Drop]
    G -->|No| I{Error Rate<br/>>20%?}
    I -->|Yes| J[Trigger: Error Rate]
    I -->|No| Z

    D --> K[Start Retraining]
    F --> K
    H --> K
    J --> K
    Z --> A
```

## Data Generation Pipeline

```mermaid
graph LR
    A[Raw Usage Logs] --> B[Filter Quality ≥0.7]
    B --> C[Filter Feedback ≥1]
    C --> D[Score with Discriminator]
    D --> E[Deduplicate by Hash]
    E --> F[Merge with Rehearsal Buffer]
    F --> G[Format Conversion]
    G --> H[train.jsonl]
    G --> I[val.jsonl]
```

## A/B Testing Flow

```mermaid
sequenceDiagram
    participant U as User
    participant R as Router
    participant C as Champion
    participant Ch as Challenger
    participant M as Metrics

    Note over R: Traffic Split: 90%/10%

    U->>R: Query
    R->>R: Random(0-1)
    alt < 0.9
        R->>C: Route to Champion
        C->>U: Response
        C->>M: Log Metrics
    else ≥ 0.9
        R->>Ch: Route to Challenger
        Ch->>U: Response
        Ch->>M: Log Metrics
    end

    Note over M: After 48 hours & 100+ samples

    M->>M: Compare Quality,<br/>Feedback, Latency

    alt Challenger 5% Better
        M->>Ch: Promote to Champion
        Note over Ch: Now serves 100%
    else Not Better
        M->>C: Keep as Champion
        Note over Ch: Retire Challenger
    end
```

## Component Interaction

```mermaid
graph TB
    subgraph "UsageLogger"
        A1[log()] --> A2[SQLite Insert]
        A3[record_feedback()] --> A2
        A4[get_records()] --> A5[Query + Filter]
    end

    subgraph "TrainingDataGenerator"
        B1[generate()] --> B2[Collect Candidates]
        B2 --> B3[Score Quality]
        B3 --> B4[Filter + Dedupe]
        B4 --> B5[Export JSONL]
    end

    subgraph "RetrainTrigger"
        C1[check_triggers()] --> C2[Sample Count?]
        C1 --> C3[Schedule?]
        C1 --> C4[Quality Drop?]
        C1 --> C5[Error Rate?]
        C2 --> C6[TriggerResult]
        C3 --> C6
        C4 --> C6
        C5 --> C6
    end

    subgraph "ABTestManager"
        D1[deploy_challenger()] --> D2[Set Traffic Split]
        D3[route_request()] --> D4[Random Selection]
        D5[compare_models()] --> D6[Compute Improvement]
        D6 --> D7[Winner?]
        D8[auto_promote()] --> D9[Promote Challenger]
    end

    subgraph "ContinuousLearningLoop"
        E1[run_iteration()] --> E2[Check Triggers]
        E2 --> E3[Generate Data]
        E3 --> E4[Train Model]
        E4 --> E5[Deploy Challenger]
        E5 --> E6[Compare & Promote]
    end

    A2 --> B2
    A5 --> C1
    C6 --> B1
    B5 --> E4
    D7 --> D8
    E2 --> C1
    E3 --> B1
    E5 --> D1
    E6 --> D5
```

## State Transitions

```mermaid
stateDiagram-v2
    [*] --> NoChampion: Initialize

    NoChampion --> Champion: Deploy First Model

    Champion --> Testing: Deploy Challenger
    Testing --> Champion: Challenger Worse
    Testing --> NewChampion: Challenger Better

    NewChampion --> Testing: Deploy New Challenger
    NewChampion --> NewChampion: No Challenger

    note right of NoChampion
        No model deployed
        100% traffic to default
    end note

    note right of Champion
        Single model
        100% traffic
    end note

    note right of Testing
        A/B test active
        90% champion
        10% challenger
    end note

    note right of NewChampion
        Promotion occurred
        Old model retired
        New model 100% traffic
    end note
```

## Time-Based Flow

```mermaid
gantt
    title Continuous Learning Timeline
    dateFormat  YYYY-MM-DD
    section Production
    Usage Logging           :a1, 2026-01-14, 7d
    Feedback Collection     :a2, 2026-01-14, 7d
    section Triggers
    Hourly Checks          :b1, 2026-01-14, 7d
    Trigger Fires          :milestone, b2, 2026-01-17, 0d
    section Retraining
    Generate Data          :c1, 2026-01-17, 1h
    Train Model            :c2, after c1, 4h
    Deploy Challenger      :c3, after c2, 10m
    section A/B Test
    10% Traffic to New     :d1, after c3, 2d
    Compare Metrics        :milestone, d2, after d1, 0d
    Promote to Champion    :d3, after d2, 10m
    section Next Cycle
    Continue Logging       :e1, after d3, 7d
```

## Quality Score Distribution

```mermaid
graph LR
    A[All Queries] -->|Quality ≥0.7| B[High Quality<br/>30%]
    A -->|Quality 0.5-0.7| C[Medium Quality<br/>50%]
    A -->|Quality <0.5| D[Low Quality<br/>20%]

    B -->|Feedback +1| E[Training Data]
    B -->|Feedback 0/-1| F[Discard]
    C -->|Feedback +1| G[Maybe Include]
    C -->|Feedback 0/-1| F
    D --> F

    E --> H[Final Dataset<br/>~10% of Total]
```

## Performance Metrics

```mermaid
graph TD
    A[Model Performance] --> B[Quality Score]
    A --> C[Feedback Rate]
    A --> D[Latency]

    B --> E{Avg Quality}
    C --> F{Positive %}
    D --> G{P95 Latency}

    E -->|Weighted 0.5| H[Overall Score]
    F -->|Weighted 0.3| H
    G -->|Weighted -0.2| H

    H --> I{Improvement<br/>≥5%?}
    I -->|Yes| J[Promote]
    I -->|No| K[Keep Testing]
```
