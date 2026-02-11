# Contextual Bandit-Based News Article Recommendation
---

## Overview

This project implements a contextual multi-armed bandit system for personalized news article recommendation. The system learns to recommend news articles from different categories (Entertainment, Education, Tech, Crime) to three distinct user segments, optimizing for engagement over time.

---

## Approach

### 1. Data Processing & Feature Engineering

**Datasets Used:**
- `news_articles.csv`: Contains 200+ news articles with headlines, categories, descriptions, and metadata
- `train_users.csv`: 2,000 user records with 33 behavioral features
- `test_users.csv`: 500 user records for testing

**Preprocessing Steps:**
- **Missing Value Handling:**
  - User age: Imputed with mean values
  - News metadata: Filled with placeholder values ("Unknown Headline", "Unknown Author")
- **Feature Encoding:**
  - Categorical features (browser_version, region_code) encoded using LabelEncoder
  - News categories encoded numerically
  - Target labels (user_1, user_2, user_3) transformed for classification

### 2. User Context Classification

**Model:** XGBoost Gradient Boosting Classifier

**Architecture:**
- Objective: Multi-class classification (3 user types)
- Training set: 1,600 samples (80%)
- Validation set: 400 samples (20%)
- Hyperparameters:
  - n_estimators: 1,200
  - learning_rate: 0.01
  - max_depth: 5
  - min_child_weight: 3
  - gamma: 0.2
  - subsample: 0.8
  - colsample_bytree: 0.8
  - Regularization: L1 (0.1), L2 (1.0)

**Performance Metrics:**
- Overall Accuracy: **88%**
- User_1: Precision 0.89, Recall 0.81, F1-Score 0.85
- User_2: Precision 0.95, Recall 0.88, F1-Score 0.92
- User_3: Precision 0.78, Recall 0.96, F1-Score 0.86

### 3. Contextual Bandit Framework

**Problem Formulation:**
- **Arms (k):** 12 total arms representing combinations of 3 user contexts × 4 news categories
- **Context:** User features predicting user segment (user_1, user_2, user_3)
- **Actions:** Recommend news article from specific category
- **Reward:** Sampled engagement score (using rlcmab_sampler)

**Arm Mapping:**
| Arm Index | User Context | News Category |
|-----------|--------------|---------------|
| 0-3       | user_1       | Entertainment, Education, Tech, Crime |
| 4-7       | user_2       | Entertainment, Education, Tech, Crime |
| 8-11      | user_3       | Entertainment, Education, Tech, Crime |

### 4. Bandit Algorithms Implemented

#### A. Epsilon-Greedy (ε = 0.1)
- **Exploration:** Random arm selection with probability ε
- **Exploitation:** Select arm with highest Q-value with probability (1-ε)
- **Update Rule:** Incremental mean: Q(a) ← Q(a) + [R - Q(a)]/N(a)

#### B. Upper Confidence Bound (UCB, C = 1.0)
- **Selection:** Maximize UCB value = Q(a) + C√(ln(t)/N(a))
- **Exploration Bonus:** Decreases as arm is selected more frequently
- **Initialization:** All unselected arms explored first

#### C. SoftMax (τ = 1.0)
- **Probabilistic Selection:** P(a) ∝ exp(Q(a)/τ)
- **Temperature Parameter:** Controls exploration-exploitation trade-off
- **Update Rule:** Same incremental mean as Epsilon-Greedy

### 5. Simulation Setup

**Parameters:**
- Time Steps (T): 10,000
- Context Selection: Random user from test set at each step
- Reward Sampling: Using rlcmab_sampler with seed 69
- Metrics Tracked: Cumulative average reward over time

---

## Results

### 1. Algorithm Performance Comparison

**Final Q-Values After 10,000 Steps:**

**Epsilon-Greedy:**
```
Arm 0-3 (user_1):   [4.13, -2.93, 2.01, 6.85]
Arm 4-7 (user_2):   [0.89, -8.14, 4.37, 3.41]
Arm 8-11 (user_3):  [5.89, -0.18, -7.37, -4.28]
```

**UCB:**
```
Arm 0-3 (user_1):   [3.70, -4.78, 2.36, 6.86]
Arm 4-7 (user_2):   [2.25, -6.23, 4.38, 3.69]
Arm 8-11 (user_3):  [5.86, -3.33, -6.39, -4.90]
```

**SoftMax:**
```
Arm 0-3 (user_1):   [4.17, -2.89, 1.89, 6.93]
Arm 4-7 (user_2):   [0.62, -7.99, 4.38, 3.44]
Arm 8-11 (user_3):  [5.89, -1.32, -6.71, -3.09]
```

**Key Observations:**
- **Best Arms Identified:**
  - User_1: Crime articles (Arm 3) with Q ≈ 6.85-6.93
  - User_2: Tech articles (Arm 6) with Q ≈ 4.37-4.38
  - User_3: Entertainment articles (Arm 8) with Q ≈ 5.86-5.89
- All three algorithms converged to similar optimal arms for each user context
- Negative Q-values indicate certain category-user combinations result in poor engagement

### 2. Hyperparameter Sensitivity Analysis

#### Epsilon-Greedy Sensitivity (ε ∈ {0.01, 0.1, 0.3})

**Findings:**
- **ε = 0.01 (Low Exploration):**
  - Fastest convergence
  - Risk of suboptimal arm selection
  - Higher short-term rewards
  
- **ε = 0.1 (Balanced):**
  - Optimal trade-off between exploration and exploitation
  - Highest cumulative rewards over 10,000 steps
  - Recommended default value
  
- **ε = 0.3 (High Exploration):**
  - Slower convergence
  - Better discovery of arm quality
  - Lower immediate rewards but potentially better long-term performance

#### UCB Sensitivity (C ∈ {0.5, 1.0, 2.0})

**Findings:**
- **C = 0.5 (Conservative Exploration):**
  - Earlier exploitation of seemingly good arms
  - Faster initial convergence
  
- **C = 1.0 (Balanced):**
  - Standard UCB performance
  - Good balance of exploration and exploitation
  
- **C = 2.0 (Aggressive Exploration):**
  - More thorough exploration of all arms
  - Higher variance in rewards
  - Better for environments with deceptive reward structures

### 3. Sample Recommendations

The system successfully generated personalized recommendations. Example outputs:

**User_1 Context:**
- All algorithms recommended **Crime** category (Arm 3)
- Example headline: "Couple Married For 68 Years Dies In Colorado Wildfire..."
- Consistent selection across all three algorithms

**User_3 Context:**
- All algorithms recommended **Entertainment** category (Arm 8)
- Example headline: "Elon Musk Teases He's A 'Wild Card' Before SNL Debut..."
- High confidence in this selection

---

## Insights & Discussion

### Algorithm Comparison

**Epsilon-Greedy:**
- Simple to implement and understand
- Consistent exploration throughout learning
- Predictable behavior with tunable exploration rate
-  Explores uniformly, even among clearly bad arms
-  Fixed exploration rate may be suboptimal as learning progresses

**UCB:**
- Principled exploration based on uncertainty
- Automatically reduces exploration as confidence increases
- No hyperparameter tuning required (C = 1.0 works well)
- Theoretical optimality guarantees
-  Can be sensitive to initialization
-  Logarithmic exploration bonus may be slow in some scenarios

**SoftMax:**
- Probabilistic selection allows for proportional exploration
- Smooth transition between exploration and exploitation
- Better suited for non-stationary environments
- Temperature parameter requires tuning
- May converge slower than UCB
- Sensitive to scale of Q-values

### Real-World Implications

1. **User Segmentation Matters:** The classifier achieved 88% accuracy in predicting user types, enabling effective contextual recommendations

2. **Category-User Affinity Patterns:**
   - User_1 strongly prefers Crime news
   - User_2 engages most with Tech content
   - User_3 favors Entertainment articles

3. **Exploration-Exploitation Trade-off:** The hyperparameter sensitivity analysis demonstrates that optimal performance requires balanced exploration (ε ≈ 0.1 for Epsilon-Greedy, C ≈ 1.0 for UCB)

4. **Algorithm Selection Guidance:**
   - Use **UCB** for unknown reward distributions (no tuning needed)
   - Use **Epsilon-Greedy** when computational simplicity is important
   - Use **SoftMax** when smooth probabilistic selection is desired

---

## Technical Implementation

**Key Libraries:**
- `numpy`, `pandas`: Data manipulation
- `scikit-learn`: Preprocessing and metrics
- `xgboost`: Gradient boosting classifier
- `matplotlib`: Visualization
- `rlcmab_sampler`: Reward simulation

**Reproducibility:**
- Random seed: 69 (for reward sampler)
- Train-test split seed: 42
- XGBoost random_state: 42

---

## Conclusion

The demonstration of the application of contextual multi-armed bandits to personalized news recommendation is clear. All three algorithms (Epsilon-Greedy, UCB, SoftMax) converged to similar optimal policies, identifying distinct news category preferences for each user segment. The 88% classification accuracy for user context prediction enabled effective personalized recommendations, with cumulative rewards improving steadily over 10,000 time steps.

The hyperparameter sensitivity analysis revealed that moderate exploration (ε = 0.1 for Epsilon-Greedy, C = 1.0 for UCB) achieves the best balance between discovering optimal arms and maximizing cumulative reward. These findings provide practical guidance for deploying bandit algorithms in real-world recommendation systems.
