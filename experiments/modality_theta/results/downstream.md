# Modality fusion ablation downstream

Generated: 2026-09-15T16:59:22+08:00

Formula: `K = θ·norm(K^T) + (1-θ)·norm(K^V)`

- θ=0.0 vision-only
- θ=0.5 equal multimodal fusion
- θ=1.0 text-only
- asd_theta1=1.0, asd_theta2=0.0 (Ψ off), ratio=0.01, w_bit=4


## theta1=1.0, theta2=0.0, target_bit=None, ratio=0.01, w_bit=4

```
|   Tasks   |Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----------|-------|----------------|-----:|-----------|---|-----:|---|-----:|
|realworldqa|Yaml   |flexible-extract|     0|exact_match|↑  |0.6654|±  |0.0171|
```


---


## theta1=1.0, theta2=0.0, target_bit=None, ratio=0.01, w_bit=4

```
| Tasks  |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|--------|------:|------|-----:|--------|---|-----:|---|------|
|mmmu_val|      0|none  |     0|mmmu_acc|↑  |0.4878|±  |   N/A|
```

- **mmmu_val** mmmu_acc,none = 0.48778

---


## theta1=1.0, theta2=0.0, target_bit=None, ratio=0.01, w_bit=4

```
|   Tasks   |Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----------|-------|----------------|-----:|-----------|---|----:|---|-----:|
|realworldqa|Yaml   |flexible-extract|     0|exact_match|↑  |0.668|±  | 0.017|
```


---


## theta1=1.0, theta2=0.0, target_bit=None, ratio=0.01, w_bit=4

```
| Tasks  |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|--------|------:|------|-----:|--------|---|-----:|---|------|
|mmmu_val|      0|none  |     0|mmmu_acc|↑  |0.4867|±  |   N/A|
```

- **mmmu_val** mmmu_acc,none = 0.48667

---


## theta1=1.0, theta2=0.0, target_bit=None, ratio=0.01, w_bit=4

```
|   Tasks   |Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----------|-------|----------------|-----:|-----------|---|-----:|---|-----:|
|realworldqa|Yaml   |flexible-extract|     0|exact_match|↑  |0.6706|±  | 0.017|
```


---


## theta1=1.0, theta2=0.0, target_bit=None, ratio=0.01, w_bit=4

```
| Tasks  |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|--------|------:|------|-----:|--------|---|-----:|---|------|
|mmmu_val|      0|none  |     0|mmmu_acc|↑  |0.4878|±  |   N/A|
```

- **mmmu_val** mmmu_acc,none = 0.48778

---


## theta1=1.0, theta2=0.0, target_bit=None, ratio=0.01, w_bit=4

```
|   Tasks   |Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----------|-------|----------------|-----:|-----------|---|-----:|---|-----:|
|realworldqa|Yaml   |flexible-extract|     0|exact_match|↑  |0.6523|±  |0.0172|
```


---


## theta1=1.0, theta2=0.0, target_bit=None, ratio=0.01, w_bit=4

```
| Tasks  |Version|Filter|n-shot| Metric |   |Value|   |Stderr|
|--------|------:|------|-----:|--------|---|----:|---|------|
|mmmu_val|      0|none  |     0|mmmu_acc|↑  | 0.48|±  |   N/A|
```

- **mmmu_val** mmmu_acc,none = 0.48000

---


## theta1=1.0, theta2=0.0, target_bit=None, ratio=0.01, w_bit=4

```
|   Tasks   |Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----------|-------|----------------|-----:|-----------|---|-----:|---|-----:|
|realworldqa|Yaml   |flexible-extract|     0|exact_match|↑  |0.6458|±  |0.0173|
```


---


## theta1=1.0, theta2=0.0, target_bit=None, ratio=0.01, w_bit=4

```
| Tasks  |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|--------|------:|------|-----:|--------|---|-----:|---|------|
|mmmu_val|      0|none  |     0|mmmu_acc|↑  |0.4822|±  |   N/A|
```

- **mmmu_val** mmmu_acc,none = 0.48222

---


## theta1=1.0, theta2=0.0, target_bit=None, ratio=0.01, w_bit=4

```
|   Tasks   |Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----------|-------|----------------|-----:|-----------|---|-----:|---|-----:|
|realworldqa|Yaml   |flexible-extract|     0|exact_match|↑  |0.6458|±  |0.0173|
```


---


## theta1=1.0, theta2=0.0, target_bit=None, ratio=0.01, w_bit=4

```
| Tasks  |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|--------|------:|------|-----:|--------|---|-----:|---|------|
|mmmu_val|      0|none  |     0|mmmu_acc|↑  |0.4833|±  |   N/A|
```

- **mmmu_val** mmmu_acc,none = 0.48333

---

