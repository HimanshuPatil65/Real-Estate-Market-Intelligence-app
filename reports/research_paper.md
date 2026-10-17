# Machine Learning Based Buyer Segmentation and Investment Profiling for Real Estate Market Intelligence

## Abstract

This analysis segments 2,000 Parcl clients using demographic, acquisition, financing, and completed-purchase information. The source data contains 10,000 property records: 7,305 completed purchases and 2,695 available listings. A K-means model produces 4 actionable buyer groups, supported by an interactive dashboard for geographic and investment-behaviour exploration.

## 1. Business problem

Parcl needs to replace broad buyer outreach with evidence-based groups that reflect how clients buy property, finance purchases, and engage across markets. The objective is to support more relevant property recommendations, improve marketing efficiency, and identify distinct investment behaviours without using protected characteristics for targeting.

## 2. Dataset and data quality

The client table was joined to sold properties on `client_id` / `client_ref`. Available listings were not treated as purchases because they do not have a buyer reference.

| Measure | Result |
|---|---:|
| Client records modelled | 2,000 |
| Property records | 10,000 |
| Completed purchases | 7,305 |
| Available listings excluded from buyer behaviour | 2,695 |
| Countries represented | 10 |
| Regions represented | 57 |

Cleaning removed exact duplicate rows, trimmed categorical labels, parsed date and currency fields, validated positive property values, and required each completed purchase to resolve to a known client. The supplied date convention was interpreted as month-first, consistent with the monthly property transaction records.

## 3. Exploratory data analysis

| Client and purchase measure | Result |
|---|---:|
| Individual clients | 94.8% |
| Investment-purpose clients | 30.8% |
| Clients applying for a loan | 36.8% |
| Median age at reporting date | 56 years |
| Age range | 24–94 years |
| Mean satisfaction score | 3.03 / 5 |
| Median completed-purchase spend per client | $1,220,893 |
| Mean completed-purchase spend per client | $1,260,375 |

The purchase measures show meaningful variation in property price, floor area, office share, and number of completed purchases. These behaviours provide the main signal separating the segments; acquisition purpose and loan use are valuable contextual descriptors but less differentiated in this sample.

## 4. Modeling methodology

The workflow uses one row per client. Derived purchase features are purchase count, total spend, average purchase price, average floor area, and office-purchase share. Client features are age, satisfaction score, client type, country, region, acquisition purpose, loan status, and referral channel.

Numeric features are median-imputed and standardised. Categorical features are mode-imputed and one-hot encoded. K-means is evaluated for K=2 through K=8 using inertia for elbow inspection and silhouette score for cohesion and separation. Hierarchical agglomerative clustering is calculated at the final K as a secondary validation check. Names, IDs, raw date of birth, and latest purchase date are excluded from the model. Gender is retained only for descriptive fairness review and is not a cluster input.

The four-group solution was selected to meet the project brief. The highest silhouette score in the automatic comparison was 0.160 at K=3, so K=4 should be treated as a business-reporting choice rather than the statistically strongest K.

The final K-means silhouette score is **0.117**. Hierarchical validation has a silhouette score of **0.061** and an adjusted Rand index of **0.239** against K-means. These modest validation values mean the segments should guide exploration and campaign design, not be treated as fixed natural categories.

### Cluster-selection diagnostics

|   cluster_count |   inertia |   silhouette_score | selected   |
|----------------:|----------:|-------------------:|:-----------|
|               2 |   16378.6 |             0.1529 | False      |
|               3 |   14965.9 |             0.1598 | False      |
|               4 |   13966.8 |             0.1174 | True       |
|               5 |   13261.4 |             0.0968 | False      |
|               6 |   12662.3 |             0.0948 | False      |
|               7 |   12199.2 |             0.0927 | False      |
|               8 |   11809.5 |             0.0917 | False      |

## 5. Segment profiles

|   cluster_id | segment                       |   clients |   average_age |   average_satisfaction |   average_purchase_count | average_total_spend   | average_purchase_price   |   average_floor_area_sqft | average_office_purchase_share   | investment_rate   | loan_rate   | company_rate   | international_rate   | top_country   | top_region   | top_referral_channel   |
|-------------:|:------------------------------|----------:|--------------:|-----------------------:|-------------------------:|:----------------------|:-------------------------|--------------------------:|:--------------------------------|:------------------|:------------|:---------------|:---------------------|:--------------|:-------------|:-----------------------|
|            1 | C1 - Premium-property buyers  |       707 |          54   |                   3.08 |                     3.56 | $1,485,979            | $420,356                 |                   1379.11 | 14.0%                           | 29.4%             | 37.3%       | 5.9%           | 21.9%                | USA           | California   | Website                |
|            2 | C2 - Apartment-focused buyers |       682 |          54.2 |                   3.06 |                     3.5  | $1,060,614            | $303,359                 |                   1012.79 | 0.0%                            | 30.5%             | 35.2%       | 5.1%           | 25.2%                | USA           | California   | Website                |
|            3 | C3 - Mixed-use buyers         |       561 |          56.5 |                   2.89 |                     3.62 | $1,115,856            | $309,199                 |                   1023.44 | 34.2%                           | 32.4%             | 38.5%       | 4.3%           | 22.8%                | USA           | California   | Website                |
|            4 | C4 - Multi-property buyers    |        50 |          63.2 |                   3.52 |                     7.32 | $2,416,603            | $332,747                 |                   1101.05 | 13.8%                           | 34.0%             | 32.0%       | 4.0%           | 14.0%                | USA           | California   | Website                |

## 6. Recommendations

- **C1 - Premium-property buyers**: Use a premium-property journey with detailed unit specifications and appointment-led viewings.
- **C2 - Apartment-focused buyers**: Prioritise apartment inventory, neighbourhood guidance, and a straightforward home-search journey.
- **C3 - Mixed-use buyers**: Show both office and residential inventory, with use-case and yield comparisons.
- **C4 - Multi-property buyers**: Offer portfolio reviews, multi-unit options, and a relationship-managed sales journey.

## 7. Implementation and governance

The Streamlit dashboard reports results as at 2025-12-01 and allows filtering by country, region, acquisition purpose, and client type. It includes cluster distribution, investor behaviour, geography, descriptive statistics, and model-quality diagnostics.

These segments describe patterns in the supplied records. They are decision-support groups, not predictions of an individual buyer's value or eligibility. Re-train the model when material new client or sales data is available, monitor stability of the selected K, and review campaign outcomes for fairness before operational use.