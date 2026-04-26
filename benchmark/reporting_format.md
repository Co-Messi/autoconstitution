# autoconstitution Benchmark Reporting Format

## Document Information
- **Version**: 1.0
- **Last Updated**: 2024
- **Purpose**: Standardized reporting format for autoconstitution benchmark results

---

## Table of Contents
1. [Executive Summary Template](#1-executive-summary-template)
2. [Detailed Results Tables](#2-detailed-results-tables)
3. [Visualization Specifications](#3-visualization-specifications)
4. [Interpretation Guide](#4-interpretation-guide)
5. [Raw Data Format](#5-raw-data-format)

---

## 1. Executive Summary Template

### 1.1 Header Section
```
================================================================================
                    SWARMRESEARCH BENCHMARK EXECUTIVE SUMMARY
================================================================================
Report ID:        [SRB-YYYY-MM-DD-XXX]
Benchmark Suite:  [autoconstitution Benchmark vX.X]
Test Date:        [YYYY-MM-DD HH:MM:SS UTC]
Duration:         [HH:MM:SS]
Environment:      [Hardware/Cloud Spec]
================================================================================
```

### 1.2 Key Metrics Overview
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         KEY PERFORMANCE INDICATORS                          │
├───────────────────────────────┬─────────────────────────────────────────────┤
│ Metric                        │ Value                                       │
├───────────────────────────────┼─────────────────────────────────────────────┤
│ Overall Score                 │ [XX.X / 100]                                │
│ Tasks Completed               │ [X / Y] ([ZZ%])                             │
│ Average Latency               │ [XXX ms]                                    │
│ Success Rate                  │ [XX.X%]                                     │
│ Resource Efficiency           │ [XX.X%]                                     │
│ Cost per 1K Operations        │ [$X.XXX]                                    │
└───────────────────────────────┴─────────────────────────────────────────────┘
```

### 1.3 Performance Summary
```
PERFORMANCE SUMMARY
─────────────────────────────────────────────────────────────────────────────
Category              Score    Grade    Status      Trend (vs baseline)
─────────────────────────────────────────────────────────────────────────────
Task Completion       [XX.X]   [A-F]    [PASS/FAIL] [↑/↓/→ X.X%]
Accuracy              [XX.X]   [A-F]    [PASS/FAIL] [↑/↓/→ X.X%]
Speed                 [XX.X]   [A-F]    [PASS/FAIL] [↑/↓/→ X.X%]
Reliability           [XX.X]   [A-F]    [PASS/FAIL] [↑/↓/→ X.X%]
Scalability           [XX.X]   [A-F]    [PASS/FAIL] [↑/↓/→ X.X%]
─────────────────────────────────────────────────────────────────────────────
```

### 1.4 Highlights Section
```
KEY FINDINGS
─────────────────────────────────────────────────────────────────────────────
✓ Strengths:
  • [Finding 1]
  • [Finding 2]
  • [Finding 3]

⚠ Areas for Improvement:
  • [Issue 1]
  • [Issue 2]

📊 Notable Observations:
  • [Observation 1]
  • [Observation 2]
─────────────────────────────────────────────────────────────────────────────
```

### 1.5 Recommendations
```
RECOMMENDATIONS
─────────────────────────────────────────────────────────────────────────────
Priority  │ Recommendation                              │ Expected Impact
──────────┼─────────────────────────────────────────────┼──────────────────
HIGH      │ [Recommendation 1]                          │ [Impact]
MEDIUM    │ [Recommendation 2]                          │ [Impact]
LOW       │ [Recommendation 3]                          │ [Impact]
──────────┴─────────────────────────────────────────────┴──────────────────
```

---

## 2. Detailed Results Tables

### 2.1 Task-Level Results Table
```
┌────────────────────────────────────────────────────────────────────────────────┐
│                           TASK-LEVEL PERFORMANCE                               │
├──────────┬─────────────┬──────────┬──────────┬──────────┬──────────┬──────────┤
│ Task ID  │ Task Name   │ Status   │ Duration │ Accuracy │ Score    │ Grade    │
├──────────┼─────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│ T001     │ [Name]      │ PASS/FAIL│ [X.XXs]  │ [XX.X%]  │ [XX.X]   │ A/B/C/D/F│
│ T002     │ [Name]      │ PASS/FAIL│ [X.XXs]  │ [XX.X%]  │ [XX.X]   │ A/B/C/D/F│
│ T003     │ [Name]      │ PASS/FAIL│ [X.XXs]  │ [XX.X%]  │ [XX.X]   │ A/B/C/D/F│
│ ...      │ ...         │ ...      │ ...      │ ...      │ ...      │ ...      │
├──────────┴─────────────┴──────────┴──────────┴──────────┴──────────┴──────────┤
│ SUMMARY              │ Pass: [X] │ Fail: [Y] │ Avg: [X.XXs] │ Avg: [XX.X%]    │
└────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Metric Breakdown Table
```
┌────────────────────────────────────────────────────────────────────────────────┐
│                          DETAILED METRIC BREAKDOWN                             │
├────────────────────┬────────────┬────────────┬────────────┬────────────────────┤
│ Metric             │ Value      │ Target     │ Threshold  │ Status             │
├────────────────────┼────────────┼────────────┼────────────┼────────────────────┤
│ Response Time (p50)│ [XXX ms]   │ [< 500ms]  │ [< 1000ms] │ [✓ PASS / ✗ FAIL]  │
│ Response Time (p95)│ [XXX ms]   │ [< 1000ms] │ [< 2000ms] │ [✓ PASS / ✗ FAIL]  │
│ Response Time (p99)│ [XXX ms]   │ [< 2000ms] │ [< 5000ms] │ [✓ PASS / ✗ FAIL]  │
│ Throughput (req/s) │ [XXXX]     │ [> 100]    │ [> 50]     │ [✓ PASS / ✗ FAIL]  │
│ Error Rate         │ [X.XX%]    │ [< 1%]     │ [< 5%]     │ [✓ PASS / ✗ FAIL]  │
│ CPU Utilization    │ [XX.X%]    │ [< 80%]    │ [< 95%]    │ [✓ PASS / ✗ FAIL]  │
│ Memory Usage       │ [X.X GB]   │ [< 8GB]    │ [< 16GB]   │ [✓ PASS / ✗ FAIL]  │
│ Token Efficiency   │ [XX.X%]    │ [> 80%]    │ [> 60%]    │ [✓ PASS / ✗ FAIL]  │
│ Cost per Request   │ [$X.XXXX]  │ [< $0.01]  │ [< $0.05]  │ [✓ PASS / ✗ FAIL]  │
└────────────────────┴────────────┴────────────┴────────────┴────────────────────┘
```

### 2.3 Comparative Analysis Table
```
┌────────────────────────────────────────────────────────────────────────────────┐
│                        COMPARATIVE ANALYSIS                                    │
├────────────────────┬─────────────┬─────────────┬─────────────┬─────────────────┤
│ Metric             │ Current     │ Baseline    │ Previous    │ Best Recorded   │
├────────────────────┼─────────────┼─────────────┼─────────────┼─────────────────┤
│ Overall Score      │ [XX.X]      │ [XX.X]      │ [XX.X]      │ [XX.X]          │
│ Avg Latency        │ [XXX ms]    │ [XXX ms]    │ [XXX ms]    │ [XXX ms]        │
│ Success Rate       │ [XX.X%]     │ [XX.X%]     │ [XX.X%]     │ [XX.X%]         │
│ Throughput         │ [XXXX rps]  │ [XXXX rps]  │ [XXXX rps]  │ [XXXX rps]      │
│ Cost Efficiency    │ [XX.X%]     │ [XX.X%]     │ [XX.X%]     │ [XX.X%]         │
├────────────────────┼─────────────┼─────────────┼─────────────┼─────────────────┤
│ Delta vs Baseline  │ [+/- X.X%]  │ ─           │ [+/- X.X%]  │ [+/- X.X%]      │
└────────────────────┴─────────────┴─────────────┴─────────────┴─────────────────┘
```

### 2.4 Error Analysis Table
```
┌────────────────────────────────────────────────────────────────────────────────┐
│                            ERROR ANALYSIS                                      │
├────────────────────┬────────────┬────────────┬─────────────────────────────────┤
│ Error Type         │ Count      │ Percentage │ Affected Tasks                  │
├────────────────────┼────────────┼────────────┼─────────────────────────────────┤
│ Timeout            │ [XX]       │ [X.X%]     │ T001, T003, T007                │
│ Validation Error   │ [XX]       │ [X.X%]     │ T002, T005                      │
│ API Error          │ [XX]       │ [X.X%]     │ T004, T006                      │
│ Resource Exhaustion│ [XX]       │ [X.X%]     │ T008, T009                      │
│ Other              │ [XX]       │ [X.X%]     │ T010                            │
├────────────────────┼────────────┼────────────┼─────────────────────────────────┤
│ TOTAL              │ [XXX]      │ [XX.X%]    │ [XX / YY tasks affected]        │
└────────────────────┴────────────┴────────────┴─────────────────────────────────┘
```

### 2.5 Resource Utilization Table
```
┌────────────────────────────────────────────────────────────────────────────────┐
│                          RESOURCE UTILIZATION                                  │
├────────────────────┬────────────┬────────────┬────────────┬────────────────────┤
│ Resource           │ Min        │ Max        │ Avg        │ Peak Time          │
├────────────────────┼────────────┼────────────┼────────────┼────────────────────┤
│ CPU (%)            │ [X.X]      │ [XX.X]     │ [XX.X]     │ [HH:MM:SS]         │
│ Memory (GB)        │ [X.X]      │ [X.X]      │ [X.X]      │ [HH:MM:SS]         │
│ GPU Memory (GB)    │ [X.X]      │ [X.X]      │ [X.X]      │ [HH:MM:SS]         │
│ Network I/O (MB/s) │ [X.X]      │ [XXX.X]    │ [XX.X]     │ [HH:MM:SS]         │
│ Disk I/O (MB/s)    │ [X.X]      │ [XX.X]     │ [X.X]      │ [HH:MM:SS]         │
│ Active Connections │ [X]        │ [XXX]      │ [XX]       │ [HH:MM:SS]         │
└────────────────────┴────────────┴────────────┴────────────┴────────────────────┘
```

---

## 3. Visualization Specifications

### 3.1 Required Visualizations

#### 3.1.1 Overall Score Gauge Chart
```
Specification:
├── Type: Gauge/Speedometer Chart
├── Dimensions: 400x300 pixels
├── Range: 0-100
├── Color Zones:
│   ├── 0-60: Red (#FF4444) - Poor
│   ├── 60-75: Yellow (#FFBB33) - Fair
│   ├── 75-85: Light Green (#00C851) - Good
│   └── 85-100: Dark Green (#007E33) - Excellent
├── Display: Current score with needle indicator
└── Labels: Score value, grade letter, status
```

#### 3.1.2 Performance Radar Chart
```
Specification:
├── Type: Radar/Spider Chart
├── Dimensions: 600x600 pixels
├── Axes (6-8 metrics):
│   ├── Task Completion
│   ├── Accuracy
│   ├── Speed/Latency
│   ├── Reliability
│   ├── Scalability
│   ├── Cost Efficiency
│   └── Resource Efficiency
├── Scale: 0-100 per axis
├── Series:
│   ├── Current Run (solid line, primary color)
│   ├── Baseline (dashed line, secondary color)
│   └── Target (dotted line, reference)
└── Legend: Bottom-right corner
```

#### 3.1.3 Latency Distribution Histogram
```
Specification:
├── Type: Histogram with KDE overlay
├── Dimensions: 800x400 pixels
├── X-axis: Response Time (ms) - logarithmic scale
├── Y-axis: Frequency / Density
├── Bins: 50 bins, auto-range based on data
├── Lines:
│   ├── p50 marker (dashed, blue)
│   ├── p95 marker (dashed, orange)
│   ├── p99 marker (dashed, red)
│   └── Target threshold (solid, green)
├── Colors:
│   ├── Bars: Light blue with dark blue border
│   └── KDE: Dark blue line
└── Annotations: Percentile values labeled
```

#### 3.1.4 Time Series Performance Chart
```
Specification:
├── Type: Multi-line Time Series
├── Dimensions: 1000x500 pixels
├── X-axis: Time (HH:MM:SS format)
├── Y-axis: Metric values (dual axis if needed)
├── Lines:
│   ├── Throughput (left axis, blue)
│   ├── Latency (right axis, orange)
│   ├── Error Rate (right axis, red)
│   └── CPU Usage (right axis, green)
├── Features:
│   ├── Interactive tooltips
│   ├── Zoom/pan capability
│   ├── Annotations for events
│   └── Legend with toggle
└── Grid: Light gray, dashed
```

#### 3.1.5 Task Completion Status Chart
```
Specification:
├── Type: Stacked Bar Chart or Treemap
├── Dimensions: 800x500 pixels
├── Categories: Task categories or individual tasks
├── Values:
│   ├── Passed (green, #28a745)
│   ├── Failed (red, #dc3545)
│   ├── Skipped (gray, #6c757d)
│   └── Timeout (orange, #fd7e14)
├── Labels: Percentage and count on segments
└── Sorting: By completion rate descending
```

#### 3.1.6 Resource Usage Timeline
```
Specification:
├── Type: Stacked Area Chart
├── Dimensions: 1000x400 pixels
├── X-axis: Time (HH:MM:SS)
├── Y-axis: Resource usage (% or GB)
├── Areas:
│   ├── CPU Usage (blue gradient)
│   ├── Memory Usage (purple gradient)
│   ├── GPU Usage (orange gradient)
│   └── Network I/O (green gradient)
├── Threshold Lines:
│   ├── Warning level (yellow dashed)
│   └── Critical level (red dashed)
└── Legend: Top-right corner
```

#### 3.1.7 Cost Breakdown Chart
```
Specification:
├── Type: Pie Chart or Donut Chart
├── Dimensions: 600x600 pixels
├── Segments:
│   ├── API Calls (color 1)
│   ├── Compute (color 2)
│   ├── Storage (color 3)
│   ├── Network (color 4)
│   └── Other (color 5)
├── Labels:
│   ├── Category name
│   ├── Dollar amount
│   └── Percentage
├── Center Text: Total Cost
└── Exploded: Largest segment
```

### 3.2 Visualization Export Formats
```
Export Requirements:
├── Primary: PNG (300 DPI minimum)
├── Secondary: SVG (scalable)
├── Interactive: HTML/JS (Plotly)
├── Data: CSV/JSON (underlying data)
└── Report: Embedded in PDF
```

### 3.3 Color Palette
```
Primary Colors:
├── Success: #28a745 (green)
├── Warning: #ffc107 (yellow)
├── Danger: #dc3545 (red)
├── Info: #17a2b8 (cyan)
└── Primary: #007bff (blue)

Secondary Colors:
├── Purple: #6f42c1
├── Orange: #fd7e14
├── Teal: #20c997
├── Pink: #e83e8c
└── Gray: #6c757d

Neutral Colors:
├── Background: #f8f9fa
├── Border: #dee2e6
├── Text: #212529
└── Muted: #868e96
```

---

## 4. Interpretation Guide

### 4.1 Scoring System

#### 4.1.1 Grade Scale
```
┌─────────────────────────────────────────────────────────────────┐
│                      GRADE INTERPRETATION                       │
├─────────┬─────────────┬─────────────────────────────────────────┤
│ Grade   │ Score Range │ Interpretation                          │
├─────────┼─────────────┼─────────────────────────────────────────┤
│ A+      │ 97-100      │ Exceptional - Exceeds all expectations  │
│ A       │ 93-96       │ Excellent - Meets all requirements      │
│ A-      │ 90-92       │ Very Good - Minor areas for improvement │
│ B+      │ 87-89       │ Good - Above average performance        │
│ B       │ 83-86       │ Above Average - Solid performance       │
│ B-      │ 80-82       │ Average - Meets basic requirements      │
│ C+      │ 77-79       │ Below Average - Some concerns           │
│ C       │ 73-76       │ Fair - Needs improvement                │
│ C-      │ 70-72       │ Poor - Significant issues present       │
│ D+      │ 67-69       │ Very Poor - Major issues                │
│ D       │ 63-66       │ Inadequate - Requires attention         │
│ D-      │ 60-62       │ Critical - Immediate action needed      │
│ F       │ 0-59        │ Failing - Unacceptable performance      │
└─────────┴─────────────┴─────────────────────────────────────────┘
```

#### 4.1.2 Status Indicators
```
┌─────────────────────────────────────────────────────────────────┐
│                      STATUS DEFINITIONS                         │
├─────────────┬───────────────────────────────────────────────────┤
│ Status      │ Definition                                        │
├─────────────┼───────────────────────────────────────────────────┤
│ PASS        │ All metrics meet or exceed target thresholds      │
│ PASS*       │ Pass with warnings - minor threshold breaches     │
│ FAIL        │ One or more critical thresholds not met           │
│ INCOMPLETE  │ Benchmark did not complete successfully           │
│ ERROR       │ Benchmark execution encountered errors            │
│ SKIPPED     │ Test was intentionally skipped                    │
└─────────────┴───────────────────────────────────────────────────┘
```

### 4.2 Metric Interpretation

#### 4.2.1 Latency Metrics
```
┌─────────────────────────────────────────────────────────────────┐
│                    LATENCY INTERPRETATION                       │
├─────────────┬───────────────┬───────────────────────────────────┤
│ Metric      │ Thresholds    │ Interpretation                    │
├─────────────┼───────────────┼───────────────────────────────────┤
│ p50 (median)│ < 200ms: Good │ Typical user experience           │
│             │ 200-500ms: OK │ Acceptable for most use cases     │
│             │ > 500ms: Poor │ May impact user satisfaction      │
├─────────────┼───────────────┼───────────────────────────────────┤
│ p95         │ < 500ms: Good │ 95% of requests are fast          │
│             │ 500-1000ms: OK│ Occasional slowdowns              │
│             │ > 1000ms: Poor│ Significant performance issues    │
├─────────────┼───────────────┼───────────────────────────────────┤
│ p99         │ < 1000ms: Good│ Tail latency acceptable           │
│             │ 1-2s: OK      │ Rare slow requests                │
│             │ > 2s: Poor    │ Worst-case needs optimization     │
└─────────────┴───────────────┴───────────────────────────────────┘
```

#### 4.2.2 Throughput Metrics
```
┌─────────────────────────────────────────────────────────────────┐
│                   THROUGHPUT INTERPRETATION                     │
├─────────────────────┬───────────────────────────────────────────┤
│ Value               │ Interpretation                            │
├─────────────────────┼───────────────────────────────────────────┤
│ > 1000 req/s        │ Excellent - High-volume capable           │
│ 500-1000 req/s      │ Good - Suitable for production            │
│ 100-500 req/s       │ Moderate - May need scaling               │
│ 50-100 req/s        │ Limited - Bottleneck likely               │
│ < 50 req/s          │ Poor - Requires optimization              │
└─────────────────────┴───────────────────────────────────────────┘
```

#### 4.2.3 Error Rate Interpretation
```
┌─────────────────────────────────────────────────────────────────┐
│                    ERROR RATE INTERPRETATION                    │
├─────────────────────┬───────────────────────────────────────────┤
│ Error Rate          │ Interpretation                            │
├─────────────────────┼───────────────────────────────────────────┤
│ < 0.1%              │ Excellent - Production ready              │
│ 0.1% - 1%           │ Good - Acceptable for most use cases      │
│ 1% - 5%             │ Fair - Monitor closely                    │
│ 5% - 10%            │ Poor - Investigation needed               │
│ > 10%               │ Critical - Immediate attention required   │
└─────────────────────┴───────────────────────────────────────────┘
```

### 4.3 Comparative Analysis Guidelines

#### 4.3.1 Delta Interpretation
```
Delta vs Baseline:
├── > +10%: Significant improvement
├── +5% to +10%: Notable improvement
├── -5% to +5%: Within normal variance
├── -10% to -5%: Notable regression
└── < -10%: Significant regression

Statistical Significance:
├── p < 0.01: Highly significant
├── p < 0.05: Significant
├── p < 0.1: Marginally significant
└── p >= 0.1: Not statistically significant
```

#### 4.3.2 Trend Analysis
```
Trend Symbols:
├── ↑ : Improvement (better than baseline)
├── ↓ : Regression (worse than baseline)
├── → : Stable (within 5% of baseline)
├── ⚠ : Warning (degraded but within threshold)
└── ✓ : Success (exceeded target)
```

### 4.4 Actionable Insights Framework

#### 4.4.1 Priority Matrix
```
┌─────────────────────────────────────────────────────────────────┐
│                      PRIORITY MATRIX                            │
├─────────────────┬───────────────────────────────────────────────┤
│ Priority        │ Action Timeline                               │
├─────────────────┼───────────────────────────────────────────────┤
│ CRITICAL        │ Immediate action (within 24 hours)            │
│ HIGH            │ Address within 1 week                         │
│ MEDIUM          │ Address within 1 month                        │
│ LOW             │ Address in next release cycle                 │
│ INFO            │ Monitor and consider for future improvements  │
└─────────────────┴───────────────────────────────────────────────┘
```

#### 4.4.2 Issue Classification
```
Issue Categories:
├── Performance: Latency, throughput, resource usage
├── Reliability: Errors, failures, stability
├── Scalability: Load handling, concurrent users
├── Cost: Resource efficiency, operational expenses
├── Quality: Accuracy, correctness, completeness
└── Security: Authentication, authorization, data protection
```

---

## 5. Raw Data Format

### 5.1 JSON Schema

#### 5.1.1 Main Benchmark Result Schema
```json
{
  "schema_version": "1.0",
  "benchmark_id": "string (UUID)",
  "benchmark_suite": "string",
  "suite_version": "string (semver)",
  "timestamp": "string (ISO 8601)",
  "duration_seconds": "number",
  
  "environment": {
    "hardware": {
      "cpu": "string",
      "cpu_cores": "number",
      "memory_gb": "number",
      "gpu": "string",
      "gpu_memory_gb": "number"
    },
    "software": {
      "os": "string",
      "python_version": "string",
      "swarm_version": "string",
      "dependencies": {
        "package_name": "version"
      }
    },
    "configuration": {
      "concurrent_workers": "number",
      "max_retries": "number",
      "timeout_seconds": "number"
    }
  },
  
  "summary": {
    "overall_score": "number (0-100)",
    "grade": "string (A+ to F)",
    "status": "string (PASS/FAIL/INCOMPLETE)",
    "tasks_completed": "number",
    "tasks_total": "number",
    "success_rate": "number (0-1)",
    "average_latency_ms": "number",
    "p50_latency_ms": "number",
    "p95_latency_ms": "number",
    "p99_latency_ms": "number",
    "total_cost_usd": "number"
  },
  
  "tasks": [
    {
      "task_id": "string",
      "task_name": "string",
      "category": "string",
      "status": "string (PASS/FAIL/SKIPPED/ERROR)",
      "start_time": "string (ISO 8601)",
      "end_time": "string (ISO 8601)",
      "duration_ms": "number",
      "latency_ms": "number",
      "accuracy": "number (0-1)",
      "score": "number (0-100)",
      "grade": "string",
      "error": {
        "type": "string",
        "message": "string",
        "stack_trace": "string"
      },
      "metrics": {
        "tokens_input": "number",
        "tokens_output": "number",
        "api_calls": "number",
        "cost_usd": "number"
      },
      "raw_output": "string"
    }
  ],
  
  "metrics": {
    "latency": {
      "min_ms": "number",
      "max_ms": "number",
      "mean_ms": "number",
      "median_ms": "number",
      "std_dev_ms": "number",
      "p50_ms": "number",
      "p75_ms": "number",
      "p90_ms": "number",
      "p95_ms": "number",
      "p99_ms": "number"
    },
    "throughput": {
      "requests_per_second": "number",
      "tokens_per_second": "number"
    },
    "errors": {
      "total_count": "number",
      "error_types": {
        "timeout": "number",
        "validation": "number",
        "api_error": "number",
        "resource_exhaustion": "number",
        "other": "number"
      }
    },
    "resources": {
      "cpu_percent": {
        "min": "number",
        "max": "number",
        "mean": "number"
      },
      "memory_gb": {
        "min": "number",
        "max": "number",
        "mean": "number"
      },
      "gpu_memory_gb": {
        "min": "number",
        "max": "number",
        "mean": "number"
      }
    },
    "cost": {
      "total_usd": "number",
      "per_request_usd": "number",
      "per_1k_tokens_usd": "number"
    }
  },
  
  "comparisons": {
    "baseline": {
      "benchmark_id": "string",
      "delta_score": "number",
      "delta_latency_percent": "number",
      "delta_success_rate_percent": "number"
    },
    "previous": {
      "benchmark_id": "string",
      "delta_score": "number",
      "delta_latency_percent": "number",
      "delta_success_rate_percent": "number"
    }
  },
  
  "metadata": {
    "run_by": "string",
    "tags": ["string"],
    "notes": "string",
    "custom_fields": {}
  }
}
```

### 5.2 CSV Export Format

#### 5.2.1 Task-Level CSV
```csv
benchmark_id,task_id,task_name,category,status,start_time,end_time,duration_ms,latency_ms,accuracy,score,grade,tokens_input,tokens_output,api_calls,cost_usd,error_type,error_message
[UUID],[T001],[Task Name],[Category],[PASS/FAIL],[ISO8601],[ISO8601],[ms],[ms],[0-1],[0-100],[A-F],[#],[#],[#],[USD],[type],[message]
```

#### 5.2.2 Metrics Time Series CSV
```csv
benchmark_id,timestamp,metric_name,metric_value,unit
timestamp,cpu_percent,45.2,percent
timestamp,memory_gb,4.5,gb
timestamp,requests_per_second,125,rps
```

### 5.3 Parquet Schema (for large datasets)
```
Recommended for datasets > 10,000 tasks:
├── Partitioning: by date (YYYY/MM/DD)
├── Compression: Snappy
├── Row groups: 10,000 rows per group
├── Columns: Same as JSON schema, flattened
└── Metadata: Benchmark ID, timestamp in footer
```

### 5.4 Data Retention Policy
```
Retention Guidelines:
├── Raw data: 90 days
├── Aggregated results: 1 year
├── Summary reports: Indefinite
├── Failed runs: 30 days
└── Archived data: Compressed after 30 days
```

### 5.5 Data Validation Rules
```
Validation Rules:
├── benchmark_id: Required, UUID format
├── timestamp: Required, valid ISO 8601
├── overall_score: 0-100 range
├── success_rate: 0-1 range
├── duration_ms: Positive number
├── cost_usd: Non-negative number
└── task.status: One of [PASS, FAIL, SKIPPED, ERROR]
```

---

## Appendix A: Report Generation Checklist

```
Pre-Generation:
[ ] Verify all benchmark runs completed
[ ] Validate raw data integrity
[ ] Check for missing or corrupted data
[ ] Confirm baseline data available

Generation:
[ ] Generate executive summary
[ ] Create all required tables
[ ] Generate all visualizations
[ ] Run comparative analysis
[ ] Apply interpretation guidelines
[ ] Add recommendations

Post-Generation:
[ ] Review for accuracy
[ ] Verify all links and references
[ ] Check formatting consistency
[ ] Validate JSON/CSV exports
[ ] Archive raw data
[ ] Distribute to stakeholders
```

## Appendix B: Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024 | Initial release |

---

*End of autoconstitution Benchmark Reporting Format*
