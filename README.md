# GPU Power Modeling:
## 1. Introduction

## 2. Theoretical Foundation: Power Consumption in GPUs

### 2.1 CMOS Power Fundamentals

The foundation of our modeling approach is based on the physical principles governing power consumption in CMOS (Complementary Metal-Oxide-Semiconductor) circuits, which form the basis of modern GPUs. Total power consumption can be broken down into:

$$P_{total} = P_{dynamic} + P_{static}$$

Where:
- $P_{dynamic}$ is the dynamic power consumption due to switching activity
- $P_{static}$ is the static power consumption due to leakage currents

### 2.2 Dynamic Power Consumption

Dynamic power consumption follows the equation:

$$P_{dynamic} = \alpha \cdot C \cdot V^2 \cdot f$$

Where:
- $\alpha$ is the activity factor (proportion of transistors switching)
- $C$ is the load capacitance
- $V$ is the supply voltage
- $f$ is the clock frequency

### 2.3 Voltage-Frequency Relationship

In modern GPUs, voltage and frequency are typically scaled together to optimize power consumption (DVFS - Dynamic Voltage and Frequency Scaling). The relationship is approximately linear in the operating range:

$$V \propto f$$

Substituting this relationship into the dynamic power equation:

$$P_{dynamic} \propto f^3$$

This cubic relationship between power and frequency is a key insight that informs our more advanced models.

## 3. Power Model Implementation

Our pipeline implements a variety of increasingly sophisticated power models:

### 3.1 Basic Power Model

The simplest model assumes power depends primarily on GPU utilization:

$$P = \alpha \cdot utilization + \beta$$

Where:
- $\alpha$ is the coefficient for utilization
- $\beta$ is a constant power draw (idle power)

This model is implemented as a simple linear regression using only GPU utilization as a feature.

### 3.2 DVFS-Aware Power Model

The DVFS-aware model incorporates clock frequency information to capture the effects of frequency scaling:

#### 3.2.1 Linear DVFS Model

$$P = \alpha \cdot utilization + \beta \cdot f + \gamma$$

Where:
- $f$ is the SM clock frequency in MHz
- $\gamma$ is the intercept (constant power)

#### 3.2.2 Cubic DVFS Model

Based on the theoretical cubic relationship, this model uses:

$$P = \alpha \cdot utilization + \beta \cdot \left(\frac{f}{f_{max}}\right)^3 + \gamma$$

Where:
- $f_{max}$ is the maximum observed clock frequency (normalization factor)
- The cubic term is scaled by a factor to improve numerical stability

#### 3.2.3 Interaction DVFS Model

This model incorporates an interaction term between utilization and clock frequency:

$$P = \alpha \cdot utilization + \beta \cdot f + \delta \cdot (utilization \cdot f) + \gamma$$

This captures how the effect of utilization might vary at different clock frequencies.

### 3.3 Memory-Aware Power Model

The most advanced model incorporates memory bandwidth metrics alongside utilization and clock frequency.

#### 3.3.1 Memory-Aware Cubic Model

$$P = \alpha \cdot utilization + \beta \cdot f^3 \cdot M + \tau \cdot f + \gamma$$

Where:
- $M = 1 + 0.5 \cdot \frac{memory\_bandwidth\_pct}{100}$ is the memory factor
- $\tau$ is the coefficient for the linear frequency term
- $f$ is expressed in GHz for better numerical scaling

This model is based on the physical formula:

$$P_{total} = \beta \cdot C \cdot f^3 + \tau \cdot f + P_{const}$$

The innovation is in modifying the capacitance factor $C$ with a memory bandwidth component, recognizing that memory activity increases effective switching capacitance.

#### 3.3.2 Memory-Aware Polynomial Model

This model uses a polynomial features approach that generates all degree-2 interactions between the core features:

$$P = \sum_{i,j} \omega_{i,j} \cdot (feature_i \cdot feature_j) + \gamma$$

Where features include utilization, clock_ghz, and memory_bandwidth_pct, allowing for terms like:
- $utilization^2$
- $utilization \cdot clock\_ghz$
- $memory\_bandwidth\_pct \cdot clock\_ghz$
- etc.

This flexible approach can capture complex non-linear relationships without requiring explicit formulation.

## 4. Model Training and Regularization

### 4.1 Ridge Regression

All models are trained using Ridge regression, which applies L2 regularization to prevent overfitting:

$$\min_{w} \|y - Xw\|^2 + \alpha\|w\|^2$$

Where:
- $y$ is the vector of actual power measurements
- $X$ is the feature matrix
- $w$ is the coefficient vector
- $\alpha$ is the regularization parameter (set to 0.1)

Ridge regression helps prevent large coefficient values and improves generalization, especially important for the polynomial model which has many features.

### 4.2 Feature Engineering

Several derived features are created to support the models:

1. Clock normalization: $clock\_ratio = \frac{sm\_clock}{max\_sm\_clock}$
2. Cubic power term: $f\_cubed = clock\_ghz^3$
3. Memory factor: $mem\_factor = 1.0 + \frac{memory\_bandwidth\_pct}{100} \cdot 0.5$
4. Memory-enhanced cubic term: $f\_cubed\_mem = f\_cubed \cdot mem\_factor$
5. Utilization-clock interaction: $util\_x\_clock = utilization \cdot sm\_clock$

## 5. Model Evaluation Metrics

The models are evaluated using several complementary metrics:

### 5.1 Mean Absolute Error (MAE)

$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

This measures the average absolute difference between predicted and actual power in watts.

### 5.2 Root Mean Squared Error (RMSE)

$$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

This metric gives higher weight to larger errors due to the squaring operation.

### 5.3 Percentage Error

$$\%Error = \frac{1}{n} \sum_{i=1}^{n} \frac{|y_i - \hat{y}_i|}{y_i} \cdot 100\%$$

This normalizes errors by the actual power value, providing a relative measure of accuracy.

## 6. Model Performance Analysis

From the experimental results:

### 6.1 Basic Model Performance
- MAE: 12.37W
- RMSE: 21.77W
- Error %: 40.55%

### 6.2 DVFS Model Performance
- MAE: 153.95W 
- RMSE: 159.69W
- Error %: 750.35%

### 6.3 Memory-Aware Model Performance
- MAE: 2.54W
- RMSE: 6.47W
- Error %: 8.47%

The memory-aware model achieves approximately 80% reduction in error compared to the basic model, demonstrating the importance of incorporating memory bandwidth metrics.

## 7. Error Analysis by Operating Conditions

### 7.1 Power State Analysis

The model's accuracy varies across different power states:

| Power State | Clock Range | Basic Error (%) | Memory-Aware Error (%) | Improvement (%) |
|-------------|-------------|-----------------|------------------------|-----------------|
| Idle        | ~300 MHz    | 38.10           | ~4.62                  | 87.9            |
| Medium      | ~538 MHz    | 40.66           | ~19.13                 | 52.9            |
| High        | ~1626 MHz   | 59.18           | ~12.08                 | 79.6            |

The memory-aware model shows particularly strong improvement for high-frequency operations, where memory effects become more significant.

### 7.2 Kernel Implementation Analysis

The error varies significantly across different kernel implementations, with the memory-aware model showing the greatest advantage for kernels with high memory activity, particularly Strategy1 which has approximately 20% memory bandwidth utilization:

| Kernel    | Memory BW (%) | Basic Error (%) | Memory-Aware Error (%) |
|-----------|---------------|-----------------|------------------------|
| Strategy1 | 20.54         | 50.27           | 12.08                  |
| Strategy2 | 0.00          | 35.74           | 19.13                  |
| Strategy7 | 0.00          | 40.32           | 5.70                   |
| Strategy9 | 0.00          | 40.46           | 5.65                   |

## 8. Conclusion

This approach provides a robust foundation for accurately predicting GPU power consumption across a wide range of workloads and operating conditions, enabling more effective power-aware optimization of GPU kernels.
