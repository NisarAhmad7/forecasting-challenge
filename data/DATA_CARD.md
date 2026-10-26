# Azure VM Workload Dataset — data-v1

## Overview

This dataset is a curated subset of the Microsoft Azure Public Dataset V2 intended for the Cloud Workload Forecasting take-home assignment.

It contains CPU utilization traces and VM metadata for **10,000 Azure virtual machines**.

The dataset preserves real VM-level workload behavior from the original Azure trace. It has not been synthetically generated.

## Dataset Contents

### `vm_cpu_readings.csv.gz`

CPU utilization measurements for the selected virtual machines.

Columns:

| Column | Description |
| --- | --- |
| `timestamp` | Timestamp in seconds relative to the beginning of the Azure trace |
| `vm_id` | Anonymized VM identifier |
| `min_cpu` | Minimum CPU utilization during the measurement interval |
| `max_cpu` | Maximum CPU utilization during the measurement interval |
| `avg_cpu` | Average CPU utilization during the measurement interval |

Each VM contains exactly **134 measurements**, sampled at **5-minute intervals**.

The dataset therefore contains:

- 10,000 VMs
- 134 measurements per VM
- 1,340,000 CPU measurement rows
- approximately 11.1 hours of continuous observations per VM

### `vm_metadata.csv.gz`

Metadata corresponding to the selected virtual machines.

Columns:

| Column | Description |
| --- | --- |
| `vm_id` | Anonymized VM identifier |
| `subscription_id` | Anonymized subscription identifier |
| `deployment_id` | Anonymized deployment identifier |
| `timestamp_vm_created` | VM creation timestamp |
| `timestamp_vm_deleted` | VM deletion timestamp |
| `max_cpu` | Maximum CPU statistic from the source VM metadata |
| `avg_cpu` | Average CPU statistic from the source VM metadata |
| `p95_max_cpu` | 95th-percentile maximum CPU statistic |
| `vm_category` | Azure VM workload category |
| `virtual_core_bucket` | Bucketed virtual CPU count |
| `memory_gb_bucket` | Bucketed VM memory in GB |

There is exactly one metadata row for every VM included in the CPU dataset.

## Selection Method

The dataset was constructed from the first three chronological CPU-reading shards of Azure Public Dataset V2.

Only VMs with complete traces across the selected observation window were eligible.

Eligible VMs contain exactly 134 CPU measurements at 5-minute intervals.

From this population, **10,000 VMs were selected deterministically using stratification by VM category and virtual-core bucket**.

Selection within each stratum used stable SHA-256-based ranking of VM identifiers, making the selection reproducible rather than dependent on random sampling state.

The resulting category distribution is:

| VM Category | VMs |
| --- | ---: |
| Delay-insensitive | 6,180 |
| Interactive | 3,091 |
| Unknown | 729 |
| **Total** | **10,000** |

## Validation

Before publication, the curated dataset was validated to ensure:

- every selected VM has exactly 134 CPU measurements;
- consecutive measurements are spaced exactly 300 seconds apart;
- no duplicate `(vm_id, timestamp)` pairs exist;
- every VM appearing in the CPU dataset has exactly one corresponding metadata record;
- the CPU and metadata VM populations match.

## Intended Use

This dataset is intended for the Cloud Workload Forecasting take-home assignment.

Candidates are expected to perform workload exploration, construct forecasting baselines and models, use appropriate temporal validation, analyze forecasting errors, and interpret the results from an operational or capacity-planning perspective.

The relatively short observation window makes this a **short-horizon workload forecasting problem**.

## Source

The data is derived from **Microsoft Azure Public Dataset V2**.

The original dataset contains anonymized VM workload traces published by Microsoft for research purposes.

This repository distributes only the curated subset required for the assignment.

## Versioning

This release is identified as **`data-v1`**.

The contents of a published dataset version should be treated as immutable so that all candidates using the same assignment version receive identical input data.

Future dataset changes should be published as a new version rather than modifying this release.
